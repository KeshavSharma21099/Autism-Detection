# Generated by Django 3.1.2 on 2020-10-25 10:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('UserProfile', '0002_remove_userprofile_organization'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userprofile',
            name='birth_date',
        ),
        migrations.AddField(
            model_name='userprofile',
            name='age',
            field=models.IntegerField(null=True),
        ),
    ]