# Langbase Discord Bot

A Discord bot powered by Anthropic's Claude AI and Langbase integration for enhanced financial analysis and conversation capabilities.

## Features

- Multi-step reasoning and planning
- Dynamic tool selection and usage
- Conversation memory and context
- Self-reflection and improvement
- Financial analysis tools
- Discord integration

## Environment Variables

The following environment variables need to be set in Railway:

- `DISCORD_BOT_TOKEN_LANGBASE`: Your Discord bot token
- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude
- `LANGBASE_API_KEY`: Your Langbase API key (optional)
- `LANGBASE_PIPE_NAME`: Your Langbase pipe name (default: 'financial-advisor')

## Deployment on Railway

1. Fork this repository
2. Create a new project in Railway
3. Connect your forked repository
4. Add the required environment variables in Railway's dashboard
5. Deploy!

Railway will automatically:
- Install dependencies from requirements.txt
- Use the Python version specified in runtime.txt
- Start the bot using the command in Procfile

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a .env file with your environment variables
5. Run the bot:
   ```bash
   python langbase_discord_bot/discord_langbase.py
   ```

## Commands

- `$ask [question]`: Ask the bot a question
- `$debug [question]`: Ask with detailed debugging information
- `$tools`: Show available tools and their descriptions
- `$ping`: Check if the bot is alive

## Support

For issues and feature requests, please open an issue in the repository.
