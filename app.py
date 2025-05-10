from typing import List, Dict, Any, Tuple
import pickle
import math
import numpy as np
import pandas as pd
import json
from flask import Flask, request, flash, render_template, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///naive_bayes.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Dataset model for database
class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    attributes = db.Column(db.Text, nullable=False)  # Stored as JSON string
    training_data = db.Column(db.Text, nullable=False)  # Stored as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'attributes': json.loads(self.attributes),
            'training_data': json.loads(self.training_data),
            'created_at': self.created_at.isoformat()
        }

# Create database tables
with app.app_context():
    db.create_all()

class StepResult:
    """Class to store results for each test step"""
    def __init__(self, 
                prediction_details: Dict[str, List[str]], 
                probabilities: Dict[str, float],
                prediction: str,
                current_data: List[Dict[str, Any]],
                knowledge_base: Dict[str, Any],
                initial_data: List[Dict[str, Any]],
                initial_knowledge_base: Dict[str, Any],
                test_instance: Dict[str, Any]):
        self.prediction_details = prediction_details
        self.probabilities = probabilities
        self.prediction = prediction
        self.current_data = current_data
        self.knowledge_base = knowledge_base
        self.initial_data = initial_data
        self.initial_knowledge_base = initial_knowledge_base
        self.test_instance = test_instance

class NaiveBayesClassifier:
    def __init__(self):
        self.probabilities = {}
        self.classes = []
        self.features = []
        self.probability_details = {}
        self.class_counts = {}
        self.data = None
        self.target_column = None
        self.initial_data = None
        self.initial_probabilities = None

    def calculate_class_probabilities(self, data: pd.DataFrame, target_column: str):
        total_instances = len(data)
        class_counts = data[target_column].value_counts()
        
        self.class_counts = class_counts
        self.classes = sorted(class_counts.index.tolist())
        
        self.probabilities['class'] = {}
        self.probability_details['class'] = {}
        
        for cls in self.classes:
            count = class_counts[cls]
            prob = count/total_instances
            self.probabilities['class'][cls] = prob
            self.probability_details['class'][cls] = f"{count}/{total_instances} = {prob:.3f}"

    def calculate_feature_probabilities(self, data: pd.DataFrame, target_column: str):
        self.features = [col for col in data.columns if col != target_column]
        feature_probabilities = {}
        feature_details = {}

        for feature in self.features:
            feature_probabilities[feature] = {}
            feature_details[feature] = {}
            
            for cls in self.classes:
                cls_data = data[data[target_column] == cls]
                feature_values = sorted(data[feature].unique())
                
                feature_probabilities[feature][cls] = {}
                feature_details[feature][cls] = {}
                
                for value in feature_values:
                    cls_value_count = len(cls_data[cls_data[feature] == value])
                    total_value_count = len(data[data[feature] == value])
                    
                    prob = cls_value_count / total_value_count if total_value_count > 0 else 0
                    
                    feature_probabilities[feature][cls][value] = prob
                    feature_details[feature][cls][value] = {
                        'prob': prob,
                        'fraction': f"{cls_value_count}/{total_value_count}"
                    }

        self.probabilities['features'] = feature_probabilities
        self.probability_details['features'] = feature_details

    def fit(self, data: pd.DataFrame, target_column: str, is_initial: bool = False):
        self.data = data.copy()
        self.target_column = target_column
        self.calculate_class_probabilities(data, target_column)
        self.calculate_feature_probabilities(data, target_column)
        
        if is_initial:
            self.initial_data = data.copy()
            self.initial_probabilities = pickle.loads(pickle.dumps(self.probability_details))

    def predict_probability(self, instance: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        class_probabilities = {}
        detail_calculations = {}
        
        for cls in self.classes:
            prior_prob = self.probabilities['class'][cls]
            detail_calculations[cls] = [
                f"P(class {cls}) = {self.probability_details['class'][cls]}"
            ]
            
            try:
                prob = prior_prob
                valid_calculation = True
                
                for feature, value in instance.items():
                    feature_prob = self.probabilities['features'][feature][cls][value]
                    
                    detail_calculations[cls].append(
                        f"P({feature}={value}|{cls}) = {self.probability_details['features'][feature][cls][value]['fraction']} = {feature_prob:.3f}"
                    )
                    
                    if feature_prob == 0:
                        valid_calculation = False
                        break
                        
                    prob *= feature_prob
                
                if valid_calculation:
                    class_probabilities[cls] = prob
                else:
                    class_probabilities[cls] = 0
                    
            except Exception as e:
                class_probabilities[cls] = 0
                
            detail_calculations[cls].append(f"Result = {class_probabilities[cls]:.6f}")
        
        total = sum(class_probabilities.values())
        if total > 0:
            percentages = {
                cls: (prob/total) * 100 
                for cls, prob in class_probabilities.items()
            }
        else:
            percentages = {cls: 0 for cls in self.classes}

        return percentages, detail_calculations

    def predict(self, instance: Dict[str, Any]) -> Tuple[str, Dict[str, float], Dict[str, List[str]]]:
        probabilities, details = self.predict_probability(instance)
        prediction = max(probabilities, key=probabilities.get)
        return prediction, probabilities, details

def process_training_data(attributes_input: str, data_input: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not attributes_input or not data_input:
        raise ValueError('All fields are required for training data')
        
    attributes = [attr.strip() for attr in attributes_input.split(',')]
    
    data = []
    for row in data_input.split('\n'):
        if row.strip():
            values = [val.strip() for val in row.split(',')]
            if len(values) != len(attributes):
                raise ValueError(f'Each row must have {len(attributes)} values')
            data.append(dict(zip(attributes, values)))
    
    if not data:
        raise ValueError('No valid training data provided')
        
    return attributes, data

def process_test_data(test_data_input: str, attributes: List[str]) -> List[Dict[str, Any]]:
    if not test_data_input:
        raise ValueError('Test data is required')
        
    test_data = []
    for row in test_data_input.split('\n'):
        if row.strip():
            values = [val.strip() for val in row.split(',')]
            if len(values) != len(attributes) - 1:
                raise ValueError(f'Each test row must have {len(attributes)-1} values')
            test_data.append(dict(zip(attributes[:-1], values)))
    
    if not test_data:
        raise ValueError('No valid test data provided')
        
    return test_data

def create_empty_knowledge_base():
    return {
        'class': {},
        'features': {}
    }

@app.route('/save_dataset', methods=['POST'])
def save_dataset():
    try:
        title = request.form.get('dataset_title')
        attributes_input = request.form.get('attributes')
        data_input = request.form.get('data')

        if not all([title, attributes_input, data_input]):
            raise ValueError('All fields are required')

        # Process the input data
        attributes = [attr.strip() for attr in attributes_input.split(',')]
        data = []
        for row in data_input.split('\n'):
            if row.strip():
                values = [val.strip() for val in row.split(',')]
                if len(values) != len(attributes):
                    raise ValueError(f'Each row must have {len(attributes)} values')
                data.append(dict(zip(attributes, values)))

        # Create new dataset
        new_dataset = Dataset(
            title=title,
            attributes=json.dumps(attributes),
            training_data=json.dumps(data)
        )
        db.session.add(new_dataset)
        db.session.commit()

        return jsonify({'success': True, 'message': 'Dataset saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/get_datasets', methods=['GET'])
def get_datasets():
    datasets = Dataset.query.all()
    return jsonify([{'id': d.id, 'title': d.title} for d in datasets])

@app.route('/get_dataset/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    return jsonify(dataset.to_dict())

@app.route('/', methods=['GET', 'POST'])
def index():
    default_context = {
        'show_training_results': False,
        'show_test_results': False,
        'attributes': [],
        'initial_data': [],
        'initial_knowledge_base': create_empty_knowledge_base(),
        'step_results': None,
        'datasets': Dataset.query.all()  # Add available datasets to context
    }
    
    if request.method == 'POST':
        try:
            if 'train' in request.form:
                dataset_id = request.form.get('dataset_id')
                
                if dataset_id:
                    # Load dataset from database
                    dataset = Dataset.query.get(dataset_id)
                    attributes = json.loads(dataset.attributes)
                    data = json.loads(dataset.training_data)
                else:
                    # Process new input data
                    attributes_input = request.form.get('attributes', '')
                    data_input = request.form.get('data', '')
                    attributes, data = process_training_data(attributes_input, data_input)

                df = pd.DataFrame(data)
                classifier = NaiveBayesClassifier()
                classifier.fit(df, attributes[-1], is_initial=True)
                
                session['attributes'] = attributes
                session['training_data'] = data
                session['classifier'] = pickle.dumps(classifier)
                
                return render_template('index.html',
                                    show_training_results=True,
                                    show_test_results=False,
                                    attributes=attributes,
                                    initial_data=data,
                                    initial_knowledge_base=classifier.probability_details,
                                    step_results=None,
                                    datasets=Dataset.query.all())
                                    
            elif 'test' in request.form:
                if 'classifier' not in session:
                    raise ValueError('Please submit training data first')
                
                attributes = session['attributes']
                training_data = session['training_data']
                classifier = pickle.loads(session['classifier'])
                
                test_data_input = request.form.get('test_data', '')
                test_instances = process_test_data(test_data_input, attributes)
                
                step_results = []
                current_data = training_data.copy()
                
                initial_classifier = pickle.loads(session['classifier'])
                
                for test_instance in test_instances:
                    prediction, probs, details = classifier.predict(test_instance)
                    
                    current_data.append({**test_instance, attributes[-1]: prediction})
                    df = pd.DataFrame(current_data)
                    classifier.fit(df, attributes[-1])
                    
                    step_result = StepResult(
                        prediction_details=details,
                        probabilities=probs,
                        prediction=prediction,
                        current_data=current_data.copy(),
                        knowledge_base=classifier.probability_details.copy(),
                        initial_data=initial_classifier.initial_data.to_dict('records'),
                        initial_knowledge_base=initial_classifier.initial_probabilities,
                        test_instance=test_instance
                    )
                    step_results.append(step_result)
                    
                session['classifier'] = pickle.dumps(classifier)
                session['training_data'] = current_data
                
                return render_template('index.html',
                                    show_training_results=True,
                                    show_test_results=True,
                                    attributes=attributes,
                                    initial_data=initial_classifier.initial_data.to_dict('records'),
                                    initial_knowledge_base=initial_classifier.initial_probabilities,
                                    step_results=step_results,
                                    datasets=Dataset.query.all())
                
        except Exception as e:
            flash(str(e))
            context = default_context.copy()
            if 'classifier' in session:
                classifier = pickle.loads(session['classifier'])
                context.update({
                    'show_training_results': True,
                    'attributes': session.get('attributes', []),
                    'initial_data': classifier.initial_data.to_dict('records'),
                    'initial_knowledge_base': classifier.initial_probabilities
                })
            return render_template('index.html', **context)
    
    return render_template('index.html', **default_context)

if __name__ == '__main__':
    app.run(debug=True)