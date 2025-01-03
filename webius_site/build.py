import os
import os
import sys
import django

# Set up the Django settings module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Adjust path to the project root
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webius_site.settings')  # Replace 'webius_site.settings' with your settings module if different
django.setup()

from pathlib import Path
from django.utils.text import slugify
from blog.models import Article
from convert import convert_markdown


def clear_database():
    """Clears all data from the database.
    """
    try:
        Article.objects.all().delete()
        print("All articles have been deleted.")
    except Exception as e:
        print(f"An error occurred while clearing the database: {e}")

def build_articles():
    """Loops over all directories in ./articles and processes files to create Articles in the database.
    """
    prev = None
    base_dir = Path('./articles')

    if not base_dir.exists():
        print("Directory './articles' does not exist.")
        return

    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Check for converted.txt
        converted_file = subdir / 'converted.txt'
        if converted_file.exists():
            with open(converted_file, 'r', encoding='utf-8') as file:
                content = file.read()
            # Use subdirectory name as title, or customize as needed
            title = subdir.name.replace("_", " ")
            slug = slugify(title)

            # Create or update the Article
            article, created = Article.objects.update_or_create(
                slug=slug,
                defaults={'title': title, 'content': content}
            )
            # if prev is not None:
            #     article.previous_article = prev
            #     article.save() 
            prev = article
            print(f"{'Created' if created else 'Updated'} Article: {title} from converted file")
            continue

        # Check for Markdown files if converted.txt does not exist
        markdown_files = list(subdir.glob('*.md'))
        if markdown_files:
            # Use the first .md file found
            md_file = markdown_files[0]
            with open(md_file, 'r', encoding='utf-8') as file:
                markdown_content = file.read()

            # Convert Markdown to HTML
            converted_content = convert_markdown(markdown_content)

            # Use subdirectory name as title, or customize as needed
            title = subdir.name.replace("_", " ")
            slug = slugify(title)

            # Create or update the Article
            article, created = Article.objects.update_or_create(
                slug=slug,
                defaults={'title': title, 'content': converted_content}
            )
            # if title == "Building a Support Vector Machine from Scratch":
            #     with open("debug_converted.txt", "w") as f:
            #         f.write(converted_content)
            
            # if prev is not None:
            #     article.previous_article = prev
            #     article.save() 
            prev = article
            print(f"{'Created' if created else 'Updated'} Article: {title} from markdown file")
        else:
            print(f"No valid files found in directory: {subdir.name}")

    # prev holds the last article created
    first_article = Article.objects.first()
    # set the last article as 'the previous' of the first 
    first_article.prev = prev
    first_article.save()


if __name__ == "__main__":
    clear_database()
    build_articles()