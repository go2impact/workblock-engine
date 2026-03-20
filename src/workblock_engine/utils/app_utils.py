"""Application-related utility functions."""
from typing import Optional


def normalize_app_name(exec_path: Optional[str]) -> str:
    """
    Extract and normalize application name from executable path.
    
    Examples:
    - "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" -> "chrome"
    - "/Applications/Slack.app/Contents/MacOS/Slack" -> "slack"
    - "C:\\Users\\john\\AppData\\Local\\slack\\slack.exe" -> "slack"
    
    Args:
        exec_path: Full path to executable
        
    Returns:
        Normalized application name
    """
    if not exec_path:
        return "unknown"
    
    # Extract filename from path
    path_parts = exec_path.replace("\\", "/").split("/")
    filename = path_parts[-1]
    
    # Remove extension and lowercase
    app_name = filename.split(".")[0].lower()
    
    # Handle common variations - map to canonical names
    app_mappings = {
        "chrome": ["chrome", "google-chrome", "chromium", "google chrome"],
        "firefox": ["firefox", "firefox-bin", "firefoxdeveloperedition"],
        "safari": ["safari", "safari technology preview"],
        "edge": ["msedge", "edge", "microsoft edge", "microsoftedge"],
        "slack": ["slack", "slackapp", "slack helper"],
        "teams": ["teams", "microsoft teams", "msteams"],
        "code": ["code", "vscode", "visual studio code", "code - insiders", "code-insiders", "electron"],
        "terminal": ["terminal", "iterm2", "iterm", "konsole", "gnome-terminal", "cmd", "powershell", "wezterm"],
        "zoom": ["zoom", "zoom.us", "zoommeeting"],
        "notion": ["notion", "notion helper"],
        "discord": ["discord", "discordcanary", "discordptb"],
        "figma": ["figma", "figma beta"],
        "obsidian": ["obsidian"],
        "outlook": ["outlook", "microsoft outlook"],
        "excel": ["excel", "microsoft excel"],
        "word": ["winword", "word", "microsoft word"],
        "powerpoint": ["powerpnt", "powerpoint", "microsoft powerpoint"],
    }
    
    # Check if app name matches any variation
    for canonical, variations in app_mappings.items():
        if app_name in variations:
            return canonical
    
    # Return cleaned app name if no mapping found
    return app_name