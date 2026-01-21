from django.db import models

# Create your models here.
class Image(models.Model):
    # caption=models.CharField(max_length=100)
    image=models.ImageField(upload_to="img/%y")
    def __str__(self):
      return self.caption
    
# Create your models here.
class Contact(models.Model):
    name = models.CharField(max_length=122)
    email = models.CharField(max_length=122)
    desc = models.TextField()
    date = models.DateField()
    
    def __str__(self):
        return self.name