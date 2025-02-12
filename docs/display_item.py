from docutils import nodes
from docutils.parsers.rst import Directive


class DisplayItemDirective(Directive):
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        "header": str,
        "description": str,
        "button_link": str,
        "col_css": str,
        "height": str,
        "tag": str,
    }

    def run(self):
        header = self.options.get("header", "")
        description = self.options.get("description", "")
        button_link = self.options.get("button_link", "#")
        col_css = self.options.get("col_css", "col-md-6")
        height = self.options.get("height", "150")
        tag = self.options.get("tag", "")

        html = f"""
        <div class="{col_css}" style="height: {height}px;">
            <div class="display-item" data-tag="{tag}">
                <h3>{header}</h3>
                <p>{description}</p>
                <a href="{button_link}" class="btn btn-primary">Learn More</a>
            </div>
        </div>
        """
        return [nodes.raw("", html, format="html")]


def setup(app):
    app.add_directive("displayitem", DisplayItemDirective)
