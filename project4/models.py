from django.db import models


class StudySession(models.Model):
    ORDER_CHOICES = [
        ('pairwise_first', 'Pairwise first'),
        ('ranking_first', 'Ranking first'),
    ]
    created_at = models.DateTimeField(auto_now_add=True)
    order = models.CharField(max_length=20, choices=ORDER_CHOICES)
    pairwise_complete = models.BooleanField(default=False)
    ranking_complete = models.BooleanField(default=False)


class PairwiseTrial(models.Model):
    session = models.ForeignKey(StudySession, on_delete=models.CASCADE, related_name='pairwise_trials')
    trial_number = models.IntegerField()
    movie_a_id = models.IntegerField()
    movie_b_id = models.IntegerField()
    chosen_movie_id = models.IntegerField()
    responded_at = models.DateTimeField(auto_now_add=True)


class RankingTrial(models.Model):
    session = models.ForeignKey(StudySession, on_delete=models.CASCADE, related_name='ranking_trials')
    round_number = models.IntegerField()
    movie_ids = models.JSONField()
    ranking = models.JSONField()
    responded_at = models.DateTimeField(auto_now_add=True)
