import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import asyncio
import logging
import io
from enhanced_agentic_chatbot import EnhancedAgenticChatbot  # Your agent file must be in the same directory

# Load environment variables from .env
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN_LANGBASE")

# Setup Discord intents and bot instance
intents = discord.Intents.default()
intents.message_content = True  # Required to receive message content in v2 bots

bot = commands.Bot(command_prefix="$", intents=intents)

# Initialize your custom agent
agent = EnhancedAgenticChatbot()

# Create a custom log handler to capture logs for Discord
class DiscordLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_messages = []
    
    def emit(self, record):
        log_entry = self.format(record)
        self.log_messages.append(log_entry)
    
    def get_logs(self):
        logs = self.log_messages.copy()
        self.log_messages.clear()
        return logs

# Add Discord log handler to the agent's logger
discord_handler = DiscordLogHandler()
discord_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'))
agent.logger.addHandler(discord_handler)

@bot.event
async def on_ready():
    print(f"🤖 Logged in as {bot.user} (ID: {bot.user.id})")
    print("✅ Langbase Discord Bot is ready and listening for commands.")

@bot.command(name="ask")
async def ask(ctx, *, question: str):
    """
    Ask the Enhanced Agentic Chatbot a question.
    Usage: $ask What do you think about AAPL?
    """
    await ctx.send("🧠 Thinking...")
    try:
        # Clear previous logs
        discord_handler.log_messages.clear()
        
        response = await agent.process_message(question)
        
        # Get captured logs
        logs = discord_handler.get_logs()
        
        # Create debug output
        debug_output = ""
        if logs:
            debug_output = "🔧 **DEBUG LOGS:**\n"
            debug_output += "```\n"
            for log in logs:
                # Only include tool usage and important logs
                if "TOOL USAGE:" in log or "Processing message" in log or "Completed agentic" in log:
                    debug_output += log + "\n"
            debug_output += "```\n"
        
        # Combine response and debug info
        full_response = f"🤖 **Response:**\n{response[:1500]}"
        if debug_output:
            full_response += f"\n\n{debug_output}"
        
        # Split if too long
        if len(full_response) > 2000:
            # Send response first
            await ctx.send(f"🤖 **Response:**\n{response[:1900]}")
            # Send debug logs separately
            if debug_output:
                await ctx.send(debug_output)
        else:
            await ctx.send(full_response)
            
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
        print(f"⚠️ Error processing message: {str(e)}")

@bot.command(name="debug")
async def debug(ctx, *, question: str):
    """
    Ask with detailed debugging information.
    Usage: $debug What do you think about AAPL?
    """
    await ctx.send("🔧 Debug mode activated...")
    try:
        # Clear previous logs
        discord_handler.log_messages.clear()
        
        response = await agent.process_message(question)
        
        # Get all captured logs
        logs = discord_handler.get_logs()
        
        # Create detailed debug output
        debug_output = "🔧 **DETAILED DEBUG LOGS:**\n"
        debug_output += "```\n"
        for log in logs:
            debug_output += log + "\n"
        debug_output += "```"
        
        # Send response and debug info in separate messages
        await ctx.send(f"🤖 **Response:**\n{response[:1900]}")
        await ctx.send(debug_output)
            
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
        print(f"⚠️ Error processing message: {str(e)}")

@bot.command(name="tools")
async def tools(ctx):
    """Show available tools and their descriptions."""
    tools_info = "🛠️ **Available Tools:**\n"
    tools_info += "```\n"
    for tool_name, tool_def in agent.available_tools.items():
        tools_info += f"📌 {tool_name.upper()}\n"
        tools_info += f"   Description: {tool_def.description}\n"
        tools_info += f"   Parameters: {tool_def.parameters}\n"
        tools_info += f"   Confidence Threshold: {tool_def.confidence_threshold}\n"
        tools_info += "\n"
    tools_info += "```"
    
    await ctx.send(tools_info)

@bot.command(name="ping")
async def ping(ctx):
    """Health check command."""
    await ctx.send("🏓 Pong! Bot is alive and responsive.")

# Main entry point
if __name__ == "__main__":
    if DISCORD_TOKEN:
        print("🔑 Discord token loaded. Starting bot...")
        bot.run(DISCORD_TOKEN)
    else:
        print("❌ DISCORD_BOT_TOKEN_LANGBASE not found in .env. Please set it first.")
