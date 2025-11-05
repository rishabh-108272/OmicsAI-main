# classification/models.py
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid

class Patient(models.Model):
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other')
    ]
    
    patient_id = models.CharField(max_length=100, unique=True, db_index=True)
    age = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(150)])
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'patients'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Patient {self.patient_id}"

class AnalysisSession(models.Model):
    MODEL_CHOICES = [
        ('colorectal_normal_vs_cancer', 'Colorectal Normal vs Cancer'),
        ('colorectal_subtype', 'Colorectal Cancer Subtype'),
        ('lung_cancer', 'Lung Cancer Classification'),
    ]
    
    session_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='analysis_sessions')
    model_type = models.CharField(max_length=50, choices=MODEL_CHOICES)
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    file_size = models.BigIntegerField()  # Added missing field
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], default='pending')
    error_message = models.TextField(blank=True, null=True)  # Added missing field
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)  # Added missing field
    
    class Meta:
        db_table = 'analysis_sessions'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Analysis {self.session_id} - {self.model_type}"

class ClassificationResult(models.Model):
    analysis_session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='results')
    result_type = models.CharField(max_length=50, choices=[
        ('normal_vs_cancer', 'Normal vs Cancer'),
        ('subtype', 'Cancer Subtype'),
        ('lung_cancer_classification', 'Lung Cancer Classification'),
    ])
    class_label = models.CharField(max_length=100)
    probability = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    confidence_score = models.FloatField(default=0.0)
    
    class Meta:
        db_table = 'classification_results'
        ordering = ['-probability']
    
    def __str__(self):
        return f"{self.class_label}: {self.probability}%"

# ADD THIS MISSING MODEL:
class ModelPerformance(models.Model):
    analysis_session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='performance_metrics')
    metric_name = models.CharField(max_length=50)
    metric_value = models.CharField(max_length=20)
    metric_description = models.TextField()
    
    class Meta:
        db_table = 'model_performance'
    
    def __str__(self):
        return f"{self.metric_name}: {self.metric_value}"
