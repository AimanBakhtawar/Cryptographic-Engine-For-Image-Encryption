from django import forms
from .models import EncryptedImage, Key

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = EncryptedImage
        fields = ['original_image']

    def save(self, commit=True):
        # Custom save method if additional logic is needed
        instance = super().save(commit=False)
        if commit:
            instance.save()
        return instance

class KeyForm(forms.ModelForm):
    class Meta:
        model = Key
        fields = ['key_value', 'key_usage']

    def save(self, user, commit=True):
        # Associate the key with the current user
        instance = super().save(commit=False)
        instance.user = user
        if commit:
            instance.save()
        return instance
