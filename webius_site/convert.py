import os
import sys
import re
import django

# Set up the Django settings module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Adjust path to the project root
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webius_site.settings')  # Replace 'webius_site.settings' with your settings module if different
django.setup()

from blog.models import Article


def convert_markdown(orig):
    code_pattern = r'```(.*?)\n(.*?)\n```'

    def replace_code_block(match):
        language = match.group(1).strip()
        code_snippet = match.group(2)
        replacement = f'<pre><code class="language-{language}">\n{code_snippet}\n</code></pre>'
        return replacement

    result = re.sub(code_pattern, replace_code_block, orig, flags=re.DOTALL)
    return result

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert.py <markdown_file>")
        sys.exit(1)

    markdown_file_path = sys.argv[1]
    directory, markdown_file = os.path.split(markdown_file_path)

    if not os.path.isfile(markdown_file_path):
        print(f"Error: File '{markdown_file}' does not exist.")
        sys.exit(1)

    with open(markdown_file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    # Convert the markdown file
    new_content = convert_markdown(markdown_content)

    converted_dir = os.path.join(directory, 'converted')
    os.makedirs(converted_dir, exist_ok=True)

    # Save the HTML file in the 'converted' directory
    output_file = os.path.join(converted_dir, os.path.splitext(markdown_file)[0] + '.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"File saved to {output_file}")

    # Save the converted Article to the db:
    title = os.path.splitext(markdown_file)[0].replace('_', ' ').capitalize()
    new_article = Article(title=title, content=new_content)
    new_article.save()
    print(f"Article '{title}' has been created and saved.")


if __name__ == "__main__":
    main()
