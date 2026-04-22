from rest_framework import serializers
from .models import Camera,Shoplifting

class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        fields = '__all__'

class ShopliftingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Shoplifting
        fields = '__all__'

    