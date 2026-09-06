from django import forms
class DatasetUploadForm(forms.Form):
    file = forms.FileField(label='Select a CSV file')

MODEL_CHOICES = [
    ('logistic', 'Logistic / Linear regression'),
    ('tree', 'Decision tree'),
    ('forest', 'Random forest'),
]

class TrainingForm(forms.Form):
    model = forms.ChoiceField(
        choices=[('linear', 'Linear / logistic regression'),
                 ('tree', 'Decision tree'),
                 ('forest', 'Random forest')])
    test_size = forms.IntegerField(min_value=10, max_value=50, initial=25,
                                   label='Percent of data held back for testing')
