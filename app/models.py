from django.db import models
from django.conf import settings


class Recipe(models.Model):
    """
    Represents a recipe with details, such as title, description, preparation time,
    and price.

    This class serves as a data model for creating and managing recipe entities,
    including attributes such as the title of the recipe, a detailed description, the
    time required to prepare it (in minutes), and its price. Instances of this class
    are typically used in applications related to recipe management or cooking guidelines.
    """
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    time_minutes = models.IntegerField()
    price = models.DecimalField(max_digits=5, decimal_places=2)

    def __str__(self):
        return self.title


class RecipeRating(models.Model):
    """
    Represents a user's rating for a specific recipe.

    This model is used to store ratings given by users to recipes. Each rating is
    associated with a recipe and a user. Ratings are expressed in stars ranging
    from 1 to 5. The model ensures that each user can only rate a recipe once,
    and ratings are ordered by creation date in descending order.
    """
    recipe = models.ForeignKey(
        Recipe, on_delete=models.CASCADE, related_name="ratings"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE
    )
    stars = models.PositiveSmallIntegerField(
        choices=[(i, f"{i} star{'s' if i>1 else ''}") for i in range(1, 6)]
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("recipe", "user")
        ordering = ["-created_at"]