from django import forms
class DatasetUploadForm(forms.Form):
    file = forms.FileField(label='Select a CSV file')