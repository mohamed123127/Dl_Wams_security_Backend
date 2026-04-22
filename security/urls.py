from rest_framework.routers import DefaultRouter
from .views import CameraViewSet,ShopliftingViewSet


router = DefaultRouter()
router.register(r'cameras', CameraViewSet, basename='cameras')
router.register(r'shoplifting', ShopliftingViewSet)

urlpatterns = router.urls