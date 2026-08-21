from django import forms
class DatasetUploadForm(forms.Form):
    file = forms.FileField(label='Select a CSV file')

MODEL_CHOICES = [
    ('logistic', 'Logistic / Linear regression'),
    ('tree', 'Decision tree'),
    ('forest', 'Random forest'),
]

class TrainingForm(forms.Form):
    model = forms.ChoiceField(choices=MODEL_CHOICES, label='Model')
    test_size = forms.IntegerField(
        min_value=10, max_value=50, initial=25)
    
