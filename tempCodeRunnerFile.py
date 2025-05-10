from typing import List, Dict, Any, Tuple
import pickle
import math
import numpy as np
import pandas as pd
from flask import Flask, request, flash, render_template, session

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

class StepResult:
    """Class to store results for each test step"""
    def __init__(self, prediction_details: Dict[str, List[str]], 
                probabilities: Dict[str, float],
                prediction: str,
                current_data: List[Dict[str, Any]],
                knowledge_base: Dict[str, Any]):
        self.prediction_details = prediction_details
        self.probabilities = probabilities
        self.prediction = prediction
        self.current_data = current_data
        self.knowledge_base = knowledge_base

class NaiveBayesClassifier:
    def __init__(self):
        self.probabilities = {}
        self.classes = []
        self.features = []
        self.probability_details = {}
        self.class_counts = {}
        self.data = None

    def calculate_class_probabilities(self, data: pd.DataFrame, target_column: str):
        """Calculate prior probabilities for each class"""
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
        """Calculate conditional probabilities for each feature value given each class"""
        self.features = [col for col in data.columns if col != target_column]
        feature_probabilities = {}
        feature_details = {}

        for feature in self.features:
            feature_probabilities[feature] = {}
            feature_details[feature] = {}
            
            for cls in self.classes:
                cls_data = data[data[target_column] == cls]
                feature_values = data[feature].unique()
                
                feature_probabilities[feature][cls] = {}
                feature_details[feature][cls] = {}
                
                for value in feature_values:
                    total_value_count = len(data[data[feature] == value])
                    cls_value_count = len(cls_data[cls_data[feature] == value])
                    
                    prob = cls_value_count / total_value_count if total_value_count > 0 else 0
                    
                    feature_probabilities[feature][cls][value] = prob
                    feature_details[feature][cls][value] = {
                        'prob': prob,
                        'fraction': f"{cls_value_count}/{total_value_count}"
                    }

        self.probabilities['features'] = feature_probabilities
        self.probability_details['features'] = feature_details

    def fit(self, data: pd.DataFrame, target_column: str):
        """Train the classifier on the given data"""
        self.data = data.copy()
        self.calculate_class_probabilities(data, target_column)
        self.calculate_feature_probabilities(data, target_column)

    def predict_probability(self, instance: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Calculate probabilities for each class for a given instance"""
        class_probabilities = {}
        detail_calculations = {}
        
        for cls in self.classes:
            prior_prob = self.probabilities['class'][cls]
            detail_calculations[cls] = [
                f"P(kelas {cls}) = {self.probability_details['class'][cls]}"
            ]
            
            try:
                # Start with log of prior probability
                log_prob = math.log(prior_prob)
                valid_calculation = True
                
                # Calculate for each feature
                for feature, value in instance.items():
                    if value not in self.probabilities['features'][feature][cls]:
                        detail_calculations[cls].append(
                            f"P({feature} = {value}|{cls}) = 0 (unknown value)"
                        )
                        valid_calculation = False
                        break
                    
                    prob = self.probabilities['features'][feature][cls][value]
                    details = self.probability_details['features'][feature][cls][value]
                    detail_calculations[cls].append(
                        f"P({feature} = {value}|{cls}) = {details['fraction']} = {prob:.3f}"
                    )
                    
                    if prob == 0:
                        valid_calculation = False
                        break
                        
                    log_prob += math.log(prob)
                
                if valid_calculation:
                    class_probabilities[cls] = math.exp(log_prob)
                else:
                    class_probabilities[cls] = 0
                    
            except Exception as e:
                class_probabilities[cls] = 0
                
            detail_calculations[cls].append(f"Hasil = {class_probabilities[cls]:.6f}")
        
        # Calculate final percentages
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
        """Make a prediction for a given instance"""
        probabilities, details = self.predict_probability(instance)
        prediction = max(probabilities, key=probabilities.get)
        return prediction, probabilities, details

def process_training_data(attributes_input: str, data_input: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Process training data input"""
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
    """Process test data input"""
    if not test_data_input:
        raise ValueError('Test data is required')
        
    test_data = []
    for row in test_data_input.split('\n'):
        if row.strip():
            values = [val.strip() for val in row.split(',')]
            if len(values) != len(attributes) - 1:  # Exclude target column
                raise ValueError(f'Each test row must have {len(attributes)-1} values')
            test_data.append(dict(zip(attributes[:-1], values)))
    
    return test_data

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main route handler"""
    if request.method == 'POST':
        try:
            if 'train' in request.form:
                # Process training data
                attributes_input = request.form.get('attributes', '')
                data_input = request.form.get('data', '')
                
                attributes, data = process_training_data(attributes_input, data_input)
                
                # Train classifier
                df = pd.DataFrame(data)
                classifier = NaiveBayesClassifier()
                classifier.fit(df, attributes[-1])
                
                # Store in session
                session['attributes'] = attributes
                session['training_data'] = data
                session['classifier'] = pickle.dumps(classifier)
                
                return render_template('index.html',
                                    show_training_results=True,
                                    attributes=attributes,
                                    initial_data=data,
                                    initial_knowledge_base=classifier.probability_details)
                
            elif 'test' in request.form:
                if 'classifier' not in session:
                    raise ValueError('Please submit training data first')
                
                # Get session data
                attributes = session['attributes']
                training_data = session['training_data']
                classifier = pickle.loads(session['classifier'])
                
                # Process test data
                test_data_input = request.form.get('test_data', '')
                test_instances = process_test_data(test_data_input, attributes)
                
                # Store results for each test instance
                step_results = []
                current_data = training_data.copy()
                
                # Process each test instance
                for test_instance in test_instances:
                    # Make prediction
                    prediction, probs, details = classifier.predict(test_instance)
                    
                    # Create result object
                    step_result = StepResult(
                        prediction_details=details,
                        probabilities=probs,
                        prediction=prediction,
                        current_data=current_data.copy(),
                        knowledge_base=classifier.probability_details.copy()
                    )
                    step_results.append(step_result)
                    
                    # Update data with prediction and retrain
                    current_data.append({**test_instance, attributes[-1]: prediction})
                    df = pd.DataFrame(current_data)
                    classifier.fit(df, attributes[-1])
                
                # Update session
                session['classifier'] = pickle.dumps(classifier)
                
                return render_template('index.html',
                                    show_training_results=True,
                                    show_test_results=True,
                                    attributes=attributes,
                                    step_results=step_results)
                
        except Exception as e:
            flash(str(e))
            return render_template('index.html',
                                show_training_results='classifier' in session,
                                attributes=session.get('attributes', []),
                                initial_data=session.get('training_data', []))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)