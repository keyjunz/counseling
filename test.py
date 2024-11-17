import torch  
from transformers import BertTokenizer  
import logging 
import argparse 
from typing import Optional, Dict  
import json 
from rich.console import Console  
from rich.markdown import Markdown  
from rich.panel import Panel  
import os
import sys
from model import Config, CounselingAI, Predictor


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s' 
)
logger = logging.getLogger(__name__)  #

# Lớp để kiểm tra mô hình tư vấn tâm lý
class CounselingTester:
    def __init__(self, model_path: str):
        self.console = Console() 
        self.config = Config()  
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)  
        
        # Tải mô hình đã huấn luyện
        self.model = self._load_model(model_path)
        self.predictor = Predictor(self.model, self.tokenizer, self.config)  

        # Tải các mẫu gợi ý (prompt templates)
        self.prompts = self._load_prompt_templates()

    def _load_model(self, model_path: str) -> CounselingAI:
        """Tải mô hình đã huấn luyện từ file checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.config.device)  # Tải trọng mô hình từ file
            model = CounselingAI(self.config) 
            model.load_state_dict(checkpoint['model_state_dict'])  
            model.eval()  
            logger.info(f"Model loaded successfully from {model_path}")  
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")  # Ghi lỗi nếu xảy ra
            raise

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Tải các mẫu gợi ý từ file JSON"""
        templates = {
            "general": "I need advice about: {question}",  # Mẫu chung
            "specific": "I'm struggling with {topic} and specifically: {question}",  # Mẫu cụ thể
            "emergency": "This is urgent and I need help with: {question}",  # Mẫu khẩn cấp
            "detailed": """  # Mẫu chi tiết
Background: {background}
Current situation: {situation}
My question: {question}
            """,
        }
        return templates

    def format_prompt(self, template_name: str, **kwargs) -> str:
        """Định dạng một mẫu gợi ý với các tham số được cung cấp"""
        if template_name not in self.prompts:
            logger.warning(f"Template {template_name} not found, using general template")  # Ghi cảnh báo nếu không tìm thấy mẫu
            template_name = "general"
        
        return self.prompts[template_name].format(**kwargs) 

    def process_response(self, response: Dict) -> None:
        """Định dạng và hiển thị phản hồi của mô hình"""
        self.console.print("\n" + "="*50)  
        
        # Hiển thị chủ đề và độ tin cậy
        self.console.print(Panel(
            f"[bold]Detected Topic:[/bold] {response['topic']}\n"
            f"[bold]Confidence:[/bold] {response['confidence']:.2%}",
            title="Analysis",
            style="blue"
        ))
        
        # Hiển thị phản hồi từ mô hình
        self.console.print(Panel(
            Markdown(response['response']),
            title="AI Response",
            style="green"
        ))
        
        self.console.print("="*50 + "\n") 

    def interactive_session(self):
        """Chạy phiên tư vấn tương tác"""
        self.console.print(Panel(
            "[bold]Welcome to the Mental Health Counseling AI Test Session[/bold]\n"
            "Type 'exit' to end the session\n"
            "Type 'templates' to see available prompt templates\n"
            "Type 'help' for more information",
            style="cyan"
        ))

        while True:
            try:
                command = self.console.input("[bold yellow]Enter command or question:[/bold yellow] ").strip()
                
                if command.lower() == 'exit':  
                    break
                elif command.lower() == 'templates': 
                    self._show_templates()
                    continue
                elif command.lower() == 'help':  
                    self._show_help()
                    continue
                
           
                if command.startswith("/template"):
                    response = self._handle_template_command(command)  
                else:
                   
                    response = self.predictor.predict(command)
                
                self.process_response(response)  
            except KeyboardInterrupt:  
                break
            except Exception as e:
                logger.error(f"Error processing input: {str(e)}")  
                self.console.print("[red]An error occurred. Please try again.[/red]")

    def _show_templates(self):
        """Hiển thị các mẫu gợi ý có sẵn"""
        self.console.print("\n[bold]Available Templates:[/bold]")
        for name, template in self.prompts.items():
            self.console.print(Panel(
                template,
                title=f"Template: {name}",
                style="blue"
            ))

    def _show_help(self):
        """Hiển thị thông tin hướng dẫn"""
        help_text = """
# Commands
- `exit`: End the session
- `templates`: Show available prompt templates
- `help`: Show this help message

# Using Templates
To use a template, type: /template <template_name> <parameters>
Example: /template specific topic=anxiety question="How do I deal with panic attacks?"

# Direct Questions
Simply type your question directly for a quick response.
        """
        self.console.print(Markdown(help_text))

    def _handle_template_command(self, command: str) -> Dict:
        """Xử lý các lệnh dựa trên mẫu gợi ý"""
        parts = command.split(maxsplit=2)
        if len(parts) < 3:
            raise ValueError("Invalid template command. Use: /template <template_name> <parameters>")
        
        template_name = parts[1]
        params_str = parts[2]
        
        # Phân tích tham số từ lệnh
        params = {}
        current_key = None
        current_value = []
        in_quotes = False
        
        for word in params_str.split():
            if '=' in word and not in_quotes:
                if current_key:
                    params[current_key] = ' '.join(current_value).strip('"')
                key, value = word.split('=', 1)
                current_key = key
                current_value = [value]
                in_quotes = value.startswith('"') and not value.endswith('"')
            else:
                current_value.append(word)
                if word.endswith('"'):
                    in_quotes = False
        
        if current_key:
            params[current_key] = ' '.join(current_value).strip('"')
        
        # Định dạng gợi ý và dự đoán
        formatted_prompt = self.format_prompt(template_name, **params)
        return self.predictor.predict(formatted_prompt)

def main():
    parser = argparse.ArgumentParser(description='Test the Mental Health Counseling AI model')  # Parser cho dòng lệnh
    parser.add_argument('--checkpoint_path', type=str, default='best_model.pt',
                    help='Path to the trained model checkpoint')
    args = parser.parse_args()

    tester = CounselingTester(args.checkpoint_path)   
    tester.interactive_session()  

if __name__ == "__main__":
    main() 
