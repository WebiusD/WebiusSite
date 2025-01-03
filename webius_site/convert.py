import os
import sys
import re
from typing import NamedTuple, Callable, List
import django

# Set up the Django settings module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Adjust path to the project root
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webius_site.settings')  # Replace 'webius_site.settings' with your settings module if different
django.setup()

from blog.models import Article

class Conversion(NamedTuple):
    pattern: str
    replacement_fn: Callable[[re.Match], str]

# the conversions applied in convert_markdown
conversions = [
    # Conversion for replacing markdown{} blocks with content inside HTML comments
    Conversion(
        pattern=r'markdown\{.*?\}\s*<!--(.*?)-->',
        replacement_fn=lambda match: match.group(1).strip()
    ),
    # Code:
    Conversion(
        pattern=r'```(.*?)\n(.*?)\n```',
        replacement_fn=lambda match: (
            f'<pre><code class="language-{match.group(1).strip()}">'
            f'{match.group(2)}\n</code></pre>'
        )
    ),
    # Headings:
    Conversion(
        pattern=r'^(#{1,6})\s*(.*?)\n',
        replacement_fn=lambda match: (
            f'<h{len(match.group(1))} style="margin-top: 20px;">{match.group(2)}</h{len(match.group(1))}>'
        )
    ),
    # Image-Links:
    Conversion(
        pattern=r'!\[(.*?)\]\((.*?)\s+"(.*?)"\)(?:\(scale=(\d*\.?\d+)\))?',
        replacement_fn=lambda match: (
            f'<figure style="text-align: center;">'
            f'<img src="/static/blog/{match.group(2)}" alt="{match.group(1)}" '
            f'{"style='width:" + str(int(float(match.group(4)) * 100)) + "%; height:auto;'" if match.group(4) else ""}>'
            f'<figcaption>{match.group(3)}</figcaption>'
            f'</figure>'
        )
    ),
    # Tables:
    Conversion(
        pattern=r'^\|(.+?)\|\n\|([:-| ]+)\|\n((\|.*\|\n)+)',
        replacement_fn=lambda match: convert_table(match.group(0))
    ),
    # double asterisk emphasis:
    Conversion(
        pattern=r'\*\*([^*]+)\*\*',
        replacement_fn=lambda match: f'<strong>{match.group(1)}</strong>'
    ),
    # single asterisk emphasis:
    Conversion(
        pattern=r'\*([^*]+)\*',
        replacement_fn=lambda match: f'<i>{match.group(1)}</i>'
    ),
    # Lists:
    Conversion(
        pattern=r'^\-\s(.*?)\n',
        replacement_fn=lambda match: f'<li>{match.group(1)}</li>'
    ),
    # Newlines:
    # Match \ followed by \n, not inside $...$ or $$...$$
    Conversion(
        pattern=r'(?<!\$)(?<!\$\$)\\\n(?!\$)(?!\$\$)',
        replacement_fn=lambda match: '<br>'
    )

    # Math
        # Conversion(
    #     pattern=r'\$(.*?)\$',
    #     replacement_fn=lambda match: f'<span class="math">{match.group(1)}</span>'
    # ),
]

def convert_table(markdown_table: str) -> str:
    """Converts a Markdown table into an HTML table, ensuring that the content of each cell
    is processed with the Markdown conversion rules.
    """
    lines = markdown_table.strip().split('\n')
    headers = lines[0].strip('|').split('|')
    separator = lines[1].strip('|').split('|')
    rows = [line.strip('|').split('|') for line in lines[2:]]

    # Build the HTML table
    html = '<table>\n<thead>\n<tr>\n'
    html += ''.join(f'<th>{convert_markdown(header.strip())}</th>' for header in headers)
    html += '</tr>\n</thead>\n<tbody>\n'
    for row in rows:
        html += '<tr>\n' + ''.join(f'<td>{convert_markdown(cell.strip())}</td>' for cell in row) + '</tr>\n'
    html += '</tbody>\n</table>'
    return html

def convert_markdown(orig):
    result = orig
    for conversion in conversions:
        result = re.sub(conversion.pattern, conversion.replacement_fn, result, flags=re.MULTILINE | re.DOTALL)

    # Wrap list items with <ul> tags
    result = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', result, flags=re.DOTALL)
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
