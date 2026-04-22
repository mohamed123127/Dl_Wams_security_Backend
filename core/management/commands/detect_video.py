from django.core.management.base import BaseCommand
from security.models import Shoplifting
from security.predicate import test_predict_video
import os
from django.conf import settings


class Command(BaseCommand):
    help = 'Run AI detection'

    def handle(self, *args, **kwargs):
        print("Running detection...")
        url =  os.path.join(settings.BASE_DIR,'security/resources/shoplifting_videos/22.mp4')
        test_predict_video(url)

        print("Done")