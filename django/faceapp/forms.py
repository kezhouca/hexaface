from django import forms

class ImageForm(forms.Form):
    imagefile = forms.ImageField(label='Select an Image to Upload:')

class DoubleImageForm(forms.Form):
    imagefile1 = forms.ImageField(label='Select a Known Image to Upload:')
    imagefile2 = forms.ImageField(label='Select a Candidate Image to Upload:')
