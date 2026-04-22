from django.db import models

# Create your models here.
class Camera(models.Model):
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=2500)
    rtsp_url = models.TextField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name
    

class Shoplifting(models.Model):
    location = models.CharField(max_length=255)
    camera = models.CharField(max_length=100)
    video_path = models.FileField(upload_to='shoplifting_videos/')
    viewed = models.BooleanField(default=False)
    
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.location} - {self.camera}"