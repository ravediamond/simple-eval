import csv
import json
import os
import uuid
from typing import List, Dict, Any, Tuple
from io import StringIO

class DatasetProcessor:
    """Handles processing and validation of dataset files"""
    
    @staticmethod
    def validate_row(row: Dict[str, Any], row_num: int) -> List[str]:
        """Validate a single row and return list of errors"""
        errors = []
        
        # Required fields
        if not row.get('question', '').strip():
            errors.append(f"Row {row_num}: 'question' is required and cannot be empty")
        
        if not row.get('reference', '').strip():
            errors.append(f"Row {row_num}: 'reference' is required and cannot be empty")
        
        # Check for excessively long fields (configurable limits)
        if len(str(row.get('question', ''))) > 10000:
            errors.append(f"Row {row_num}: 'question' too long (max 10000 characters)")
        
        if len(str(row.get('reference', ''))) > 10000:
            errors.append(f"Row {row_num}: 'reference' too long (max 10000 characters)")
        
        return errors
    
    @staticmethod
    def normalize_row(row: Dict[str, Any], row_num: int) -> Dict[str, Any]:
        """Normalize a row to standard format"""
        normalized = {
            'id': row.get('id', f"auto_{row_num}"),
            'question': str(row.get('question', '')).strip(),
            'reference': str(row.get('reference', '')).strip()
        }
        return normalized
    
    @staticmethod
    def process_csv(content: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Process CSV content and return normalized rows and errors"""
        rows = []
        errors = []
        
        try:
            # Try to detect delimiter
            sample = content[:1024]
            sniffer = csv.Sniffer()
            try:
                delimiter = sniffer.sniff(sample).delimiter
            except:
                delimiter = ','
            
            reader = csv.DictReader(StringIO(content), delimiter=delimiter)
            
            # Check required columns
            if not reader.fieldnames:
                errors.append("CSV file appears to be empty or invalid")
                return rows, errors
            
            required_columns = {'question', 'reference'}
            available_columns = set(reader.fieldnames)
            missing_columns = required_columns - available_columns
            
            if missing_columns:
                errors.append(f"Missing required columns: {', '.join(missing_columns)}")
                return rows, errors
            
            # Process rows
            for row_num, row in enumerate(reader, 1):
                # Validate row
                row_errors = DatasetProcessor.validate_row(row, row_num)
                errors.extend(row_errors)
                
                if not row_errors:  # Only add valid rows
                    normalized = DatasetProcessor.normalize_row(row, row_num)
                    rows.append(normalized)
        
        except Exception as e:
            errors.append(f"Error processing CSV: {str(e)}")
        
        return rows, errors
    
    @staticmethod
    def process_jsonl(content: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Process JSONL content and return normalized rows and errors"""
        rows = []
        errors = []
        
        try:
            lines = content.strip().split('\n')
            
            for row_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                
                try:
                    row = json.loads(line)
                    
                    # Validate row
                    row_errors = DatasetProcessor.validate_row(row, row_num)
                    errors.extend(row_errors)
                    
                    if not row_errors:  # Only add valid rows
                        normalized = DatasetProcessor.normalize_row(row, row_num)
                        rows.append(normalized)
                
                except json.JSONDecodeError as e:
                    errors.append(f"Row {row_num}: Invalid JSON - {str(e)}")
        
        except Exception as e:
            errors.append(f"Error processing JSONL: {str(e)}")
        
        return rows, errors
    
    @staticmethod
    def process_file(content: str, filename: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Process uploaded file content based on extension"""
        _, ext = os.path.splitext(filename.lower())
        
        if ext == '.csv':
            return DatasetProcessor.process_csv(content)
        elif ext == '.jsonl':
            return DatasetProcessor.process_jsonl(content)
        else:
            return [], [f"Unsupported file type: {ext}. Only .csv and .jsonl files are supported."]
    
    @staticmethod
    def check_duplicate_ids(rows: List[Dict[str, Any]]) -> List[str]:
        """Check for duplicate IDs in the dataset"""
        ids = [row['id'] for row in rows]
        seen = set()
        duplicates = set()
        
        for id_val in ids:
            if id_val in seen:
                duplicates.add(id_val)
            seen.add(id_val)
        
        if duplicates:
            return [f"Duplicate IDs found: {', '.join(duplicates)}"]
        
        return []
    
    @staticmethod
    def save_normalized_data(rows: List[Dict[str, Any]], dataset_id: int, version_number: int) -> str:
        """Save normalized data as JSONL and return file path"""
        os.makedirs("data/datasets", exist_ok=True)
        
        filename = f"dataset_{dataset_id}_v{version_number}.jsonl"
        file_path = os.path.join("data/datasets", filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
        return file_path