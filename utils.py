from IPython.display import HTML, display

color_start = '<span style="background-color: #cfe0e8;">'
color_end = '</span>'

def create_html_text(long_text, unique_text):
    """
    Creates an HTML object that displays the long_text followed by the unique_text
    in a visually distinct way.
    """
    # Replace newline characters with HTML line break tags
    long_text = long_text.replace('\n', '<br>')
    unique_text = unique_text.replace('\n', '<br>')
    
    # Add a non-breaking space between the two parts
    if not long_text.endswith('<br>'):
        long_text += '&nbsp;'
    
    # Create the HTML code
    # html = f'<div style="display: inline-block; border-radius: 3px; padding: 1px 3px;"><span>{long_text}</span><span style="background-color: #cfe0e8;">{unique_text}</span></div>'
    html = f'<div style="display: inline-block; font-size: 0.9em; line-height: 1.2em; border-radius: 3px; padding: 1px 3px;"><span>{long_text}</span><span style="background-color: #cfe0e8;">{unique_text}</span></div>'

    return html

def create_html(text):
    """
    Creates an HTML object that displays the long_text followed by the unique_text
    in a visually distinct way.
    """
    # Replace newline characters with HTML line break tags
    text = text.replace('\n', '<br>')
    
    # Create the HTML code
    html = f'<html> <body> <p style=" font-size: 13px; border-left:1em solid transparent; border-right:1em solid transparent; padding-top: 1em; "> {text} </p> </body> </html>'

    return html
