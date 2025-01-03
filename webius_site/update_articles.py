import os
import sys
import django

# Set up the Django settings module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Adjust path to the project root
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webius_site.settings')  # Replace 'webius_site.settings' with your settings module if different
django.setup()

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from pathlib import Path
import argparse

from django.utils.text import slugify
from blog.models import Article
from convert import convert_markdown

llm = ChatOpenAI(model='gpt-4o-mini')
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", """\
You create a short summary for blog posts.
"""),
    ("user", """\
Consider the following blog post. Summarize it in an enjoyable and amusing manner. Be concise do not write more than 50 words:
{content}
""")
])

def clear_database():
    """Clears all data from the database.
    """
    try:
        Article.objects.all().delete()
        print("All articles have been deleted.")
    except Exception as e:
        print(f"An error occurred while clearing the database: {e}")

def update(debug):
    """Loops over all directories in ./articles and processes files to create/update the Articles in the database.
    """
    last_article = None
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
            # If description is empty or None, generate it
            if not article.description:
                description = get_llm_summary(content)
                article.description = description
                article.save() 
            last_article = article
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
            if not article.description:
                description = get_llm_summary(content)
                article.description = description
                article.save() 
            if debug:
                with open(f"articles/debug/debug_{title}.txt", "w") as f:
                    f.write(converted_content)

            last_article = article
            print(f"{'Created' if created else 'Updated'} Article: {title} from markdown file")
        else:
            print(f"No valid files found in directory: {subdir.name}")

    first_article = Article.objects.first()
    # set the last article as 'the previous' of the first 
    first_article.prev = last_article
    first_article.save()

def get_llm_summary(content):
    response = (summarize_prompt | llm).invoke({"content": content})
    return response.content

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Build articles for the Django application.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode to save converted content to debug files.")
    args = parser.parse_args()

    # Clear the database and build articles
    # clear_database()
    update(debug=args.debug)