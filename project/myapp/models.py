from django.db import models
from django.contrib.auth.models import User

class EncryptedImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    original_image = models.ImageField(upload_to='original_images/', null=True, blank=True)
    encrypted_image = models.ImageField(upload_to='encrypted_images/', null=True, blank=True)
    encrypted_image_url = models.URLField(max_length=500, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    key = models.CharField(max_length=255, null=True, blank=True)
    shuffle_indices = models.JSONField(null=True, blank=True)
    key_value = models.IntegerField(null=True, blank=True)
    original_size = models.CharField(max_length=100, null=True, blank=True)
    otp_code = models.CharField(max_length=6, blank=True, null=True)
    otp_verified = models.BooleanField(default=False)


    def save(self, *args, **kwargs):
        if self.encrypted_image:
            self.encrypted_image_url = self.encrypted_image.url
        super(EncryptedImage, self).save(*args, **kwargs)

    def __str__(self):
        return f"Encrypted: {self.encrypted_image.name if self.encrypted_image else 'Not Encrypted'}"

class AttackResult(models.Model):
    image = models.ForeignKey(EncryptedImage, on_delete=models.CASCADE)
    attack_type = models.CharField(max_length=50)
    result_value = models.FloatField()
    attacked_on = models.DateTimeField(auto_now_add=True)
    modified_key_used = models.CharField(max_length=255, null=True, blank=True)
    pixel_coordinates = models.CharField(max_length=50, null=True, blank=True)
    original_pixel = models.IntegerField(null=True, blank=True)
    modified_pixel = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.attack_type} on {self.image}"  