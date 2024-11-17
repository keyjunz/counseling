import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from typing import Optional, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import wandb
# Load data
df = pd.read_parquet("hf://datasets/mpingale/mental-health-chat-dataset/data/train-00000-of-00001-991edb316b3098d3.parquet")
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# Save splits
train_df.to_parquet("train.parquet")
val_df.to_parquet("val.parquet")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.max_len = 512
        self.batch_size = 8
        self.num_epochs = 5
        self.learning_rate = 2e-5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Define common mental health topics as categories
        self.topics = ['anxiety', 'depression', 'relationships', 'stress', 'trauma', 
                      'addiction', 'family', 'general', 'self-esteem', 'other']

class MentalHealthDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, config: Config, split: str = 'train'):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Load and preprocess the data
        self.data = pd.read_parquet(data_path)
        # Convert topics to numeric labels
        self.topic_to_idx = {topic: idx for idx, topic in enumerate(config.topics)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Combine question title and text for fuller context
        question = f"{row['questionTitle']} {row['questionText']}"
        answer = row['answerText']
        
        # Tokenize input
        encoding = self.tokenizer(
            question,
            answer,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_len,
            return_tensors="pt"
        )
        
        # Get topic label
        topic_label = self.topic_to_idx.get(row['topic'].lower(), 
                                          len(self.config.topics) - 1)  # 'other' as default
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'topic_label': torch.tensor(topic_label),
            'labels': encoding['input_ids'].squeeze().clone()
        }

class CounselingAI(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(config.model_name)
        
        # Topic classification head
        self.topic_classifier = nn.Linear(self.bert.config.hidden_size, len(config.topics))
        
        # Response generation head
        self.response_generator = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        
        # Loss functions
        self.topic_criterion = nn.CrossEntropyLoss()
        self.generation_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
    def forward(self, input_ids, attention_mask, token_type_ids, 
                topic_label=None, labels=None):
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Topic classification
        topic_logits = self.topic_classifier(outputs.pooler_output)
        
        # Response generation
        generation_logits = self.response_generator(outputs.last_hidden_state)
        
        results = {
            'topic_logits': topic_logits,
            'generation_logits': generation_logits
        }
        
        if topic_label is not None and labels is not None:
            # Calculate losses
            topic_loss = self.topic_criterion(topic_logits, topic_label)
            generation_loss = self.generation_criterion(
                generation_logits.view(-1, self.bert.config.vocab_size),
                labels.view(-1)
            )
            
            # Combined loss
            results['loss'] = topic_loss + generation_loss
            
        return results

class Trainer:
    def __init__(self, model: CounselingAI, train_loader: DataLoader, 
                 val_loader: DataLoader, config: Config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize wandb
        wandb.init(project="mental-health-counseling", config=vars(config))
        
    def train(self):
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                progress_bar.set_postfix({'loss': loss.item()})
                wandb.log({'train_loss': loss.item()})
            
            avg_train_loss = train_loss / train_steps
            
            # Validation
            val_loss = self.evaluate()
            
            # Log metrics
            wandb.log({
                'epoch': epoch + 1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': val_loss,
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pt')
            
            logger.info(f'Epoch {epoch+1}:')
            logger.info(f'Average train loss: {avg_train_loss:.4f}')
            logger.info(f'Average validation loss: {val_loss:.4f}')
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs['loss'].item()
                total_steps += 1
        
        return total_loss / total_steps
    
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

class Predictor:
    def __init__(self, model: CounselingAI, tokenizer, config: Config):
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        
    def predict(self, question: str) -> Dict:
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            question,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_len
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get predictions
        topic_pred = torch.argmax(outputs['topic_logits'], dim=-1)
        response_ids = torch.argmax(outputs['generation_logits'], dim=-1)
        
        # Decode response
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        
        return {
            'response': response,
            'topic': self.config.topics[topic_pred.item()],
            'confidence': F.softmax(outputs['topic_logits'], dim=-1).max().item()
        }

def main():
    # Load configuration
    config = Config()
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    
    # Prepare datasets
    train_dataset = MentalHealthDataset('train.parquet', tokenizer, config, 'train')
    val_dataset = MentalHealthDataset('val.parquet', tokenizer, config, 'val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model
    model = CounselingAI(config)
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train model
    trainer.train()
    
    

if __name__ == "__main__":
    main()
