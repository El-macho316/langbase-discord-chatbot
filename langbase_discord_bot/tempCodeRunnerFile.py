import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from dotenv import load_dotenv
import requests
from anthropic import Anthropic
import sys
from dataclasses import dataclass
from enum import Enum

# Official Langbase SDK
try:
    from langbaseai import Langbase
except ImportError:
    print("⚠️  Langbase SDK not installed. Install with: pip install langbaseai")
    Langbase = None

# Add parent directory to path to import test financial function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from test_financial_function import TestFinancialDataService

# Load environment variables
load_dotenv()

class AgentState(Enum):
    """States for the agentic system"""
    LISTENING = "listening"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"

@dataclass
class ToolDefinition:
    """Definition of available tools"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    confidence_threshold: float = 0.7

@dataclass
class Memory:
    """Memory structure for conversations"""
    content: str
    timestamp: datetime
    context: Dict[str, Any]
    importance: float = 0.5

class EnhancedAgenticChatbot:
    """
    An enhanced agentic AI chatbot that properly integrates Langbase
    with advanced reasoning, memory, and tool discovery capabilities
    """
    
    def __init__(self):
        # Initialize Langbase (proper integration)
        self.langbase_api_key = os.getenv('LANGBASE_API_KEY')
        self.langbase_pipe_name = os.getenv('LANGBASE_PIPE_NAME', 'financial-advisor')
        
        # Initialize Langbase client
        self.langbase_client = None
        if Langbase and self.langbase_api_key:
            self.langbase_client = Langbase(api_key=self.langbase_api_key)
        
        # Initialize Anthropic Claude API
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.claude_client = Anthropic(api_key=self.anthropic_api_key)
        
        # Initialize financial service
        self.financial_service = TestFinancialDataService()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Agentic system components
        self.current_state = AgentState.LISTENING
        self.conversation_memory: List[Memory] = []
        self.long_term_memory: Dict[str, Any] = {}
        self.available_tools: Dict[str, ToolDefinition] = {}
        self.reasoning_chain: List[Dict[str, Any]] = []
        
        # Initialize tools
        self._initialize_tools()
        
        # Enhanced system prompt for agentic behavior
        self.system_prompt = """You are an advanced agentic AI financial analyst with the following capabilities:

CORE ABILITIES:
1. **Multi-step Reasoning**: Break complex problems into smaller steps
2. **Tool Discovery**: Dynamically select and use appropriate tools
3. **Memory Management**: Remember context across conversations
4. **Self-Reflection**: Evaluate your own responses and improve
5. **Planning**: Create and execute multi-step plans

AVAILABLE TOOLS:
- Stock analysis with comprehensive metrics
- Market comparison and ranking
- Portfolio recommendations
- Risk assessments
- Financial education

AGENTIC BEHAVIOR:
- Always think through problems step by step
- Use tools when needed, not just when explicitly asked
- Remember user preferences and previous conversations
- Reflect on the quality of your responses
- Ask clarifying questions when needed
- Provide actionable insights with proper disclaimers

REASONING PROCESS:
1. Understand the user's request
2. Break it into sub-tasks if complex
3. Select appropriate tools
4. Execute tasks in logical order
5. Synthesize results
6. Reflect on response quality
7. Provide comprehensive answer

Remember: This is for educational purposes only. Always include appropriate disclaimers."""

        self.logger.info("Enhanced Agentic Chatbot initialized successfully")

    def _setup_logger(self) -> logging.Logger:
        """Setup enhanced logger with tool usage tracking"""
        logger = logging.getLogger("EnhancedAgenticChatbot")
        if not logger.handlers:
            # Create file handler for tool usage logs
            file_handler = logging.FileHandler('tool_usage.log')
            file_handler.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        logger.setLevel(logging.INFO)
        return logger

    def _log_tool_usage(self, tool_name: str, parameters: Dict[str, Any], result: Dict[str, Any], user_context: str = ""):
        """Log tool usage with detailed information"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'tool_name': tool_name,
            'parameters': parameters,
            'success': result.get('success', False),
            'user_context': user_context,
            'result_summary': str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
        }
        
        self.logger.info(f"TOOL USAGE: {json.dumps(log_entry, indent=2)}")
        
        # Also log to a dedicated tool usage file
        with open('discord_tool_usage.log', 'a') as f:
            f.write(f"{datetime.now().isoformat()} - TOOL: {tool_name} - PARAMS: {parameters} - SUCCESS: {result.get('success', False)} - USER: {user_context}\n")

    def _initialize_tools(self):
        """Initialize available tools for the agent"""
        self.available_tools = {
            "analyze_stock": ToolDefinition(
                name="analyze_stock",
                description="Analyze a single stock with comprehensive financial metrics",
                function=self.analyze_stock_function,
                parameters={"ticker": {"type": "string", "description": "Stock ticker symbol"}},
                confidence_threshold=0.8
            ),
            "compare_stocks": ToolDefinition(
                name="compare_stocks",
                description="Compare multiple stocks and provide rankings",
                function=self.compare_stocks_function,
                parameters={"tickers": {"type": "array", "description": "List of ticker symbols", "optional": True}},
                confidence_threshold=0.7
            ),
            "get_available_tickers": ToolDefinition(
                name="get_available_tickers",
                description="Get list of available stock tickers for analysis",
                function=self.get_available_tickers_function,
                parameters={},
                confidence_threshold=0.9
            ),
            "risk_assessment": ToolDefinition(
                name="risk_assessment",
                description="Assess investment risk for a stock or portfolio",
                function=self.risk_assessment_function,
                parameters={"ticker": {"type": "string", "description": "Stock ticker symbol"}},
                confidence_threshold=0.6
            ),
            "portfolio_recommendation": ToolDefinition(
                name="portfolio_recommendation",
                description="Generate portfolio recommendations based on risk tolerance",
                function=self.portfolio_recommendation_function,
                parameters={"risk_level": {"type": "string", "description": "Risk level: low, medium, high"}},
                confidence_threshold=0.5
            )
        }

    def analyze_stock_function(self, ticker: str) -> Dict[str, Any]:
        """Enhanced stock analysis function"""
        try:
            self.logger.info(f"Analyzing stock: {ticker}")
            result = self.financial_service.analyze_stock(ticker)
            
            # Log tool usage
            self._log_tool_usage(
                tool_name="analyze_stock",
                parameters={"ticker": ticker},
                result=result,
                user_context=f"Stock analysis request for {ticker}"
            )
            
            if result['success']:
                # Add to memory
                self._add_to_memory(f"Analyzed {ticker}", {
                    'action': 'stock_analysis',
                    'ticker': ticker,
                    'score': result['data']['score'],
                    'valuation': result['data']['valuation']
                }, importance=0.8)
                
                return {
                    'success': True,
                    'ticker': ticker,
                    'report': result['data']['userFriendlyReport'],
                    'score': result['data']['score'],
                    'valuation': result['data']['valuation'],
                    'company_name': result['data']['companyName'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'success': False, 'error': result['error']}
                
        except Exception as e:
            self.logger.error(f"Error in stock analysis: {str(e)}")
            return {'success': False, 'error': f"Analysis failed: {str(e)}"}

    def risk_assessment_function(self, ticker: str) -> Dict[str, Any]:
        """Assess investment risk for a stock"""
        try:
            analysis = self.financial_service.analyze_stock(ticker)
            if not analysis['success']:
                result = {'success': False, 'error': 'Could not analyze stock for risk assessment'}
                self._log_tool_usage(
                    tool_name="risk_assessment",
                    parameters={"ticker": ticker},
                    result=result,
                    user_context=f"Risk assessment request for {ticker}"
                )
                return result
            
            data = analysis['data']
            
            # Calculate risk score based on various metrics
            pe_ratio = data.get('PE_ratio', 20)
            roe = data.get('ROE', 10)
            ev_ebitda = data.get('EV_EBITDA', 15)
            
            # Risk scoring logic
            risk_score = 0
            risk_factors = []
            
            if pe_ratio > 30:
                risk_score += 2
                risk_factors.append("High P/E ratio indicates potential overvaluation")
            elif pe_ratio < 10:
                risk_score += 1
                risk_factors.append("Very low P/E ratio may indicate underlying issues")
            
            if roe < 5:
                risk_score += 2
                risk_factors.append("Low ROE indicates poor profitability")
            
            if ev_ebitda > 20:
                risk_score += 1
                risk_factors.append("High EV/EBITDA ratio suggests high valuation")
            
            risk_level = "Low" if risk_score <= 1 else "Medium" if risk_score <= 3 else "High"
            
            result = {
                'success': True,
                'ticker': ticker,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'recommendation': f"Based on analysis, {ticker} has {risk_level.lower()} investment risk."
            }
            
            # Log tool usage
            self._log_tool_usage(
                tool_name="risk_assessment",
                parameters={"ticker": ticker},
                result=result,
                user_context=f"Risk assessment request for {ticker}"
            )
            
            return result
            
        except Exception as e:
            result = {'success': False, 'error': f"Risk assessment failed: {str(e)}"}
            self._log_tool_usage(
                tool_name="risk_assessment",
                parameters={"ticker": ticker},
                result=result,
                user_context=f"Risk assessment request for {ticker}"
            )
            return result

    def portfolio_recommendation_function(self, risk_level: str = "medium") -> Dict[str, Any]:
        """Generate portfolio recommendations"""
        try:
            all_stocks = self.financial_service.analyze_all_stocks()
            recommendations = []
            
            risk_level = risk_level.lower()
            
            for ticker, analysis in all_stocks.items():
                if analysis['success']:
                    risk_assessment = self.risk_assessment_function(ticker)
                    if risk_assessment['success']:
                        stock_risk = risk_assessment['risk_level'].lower()
                        
                        # Match stocks to risk tolerance
                        if risk_level == "low" and stock_risk in ["low"]:
                            recommendations.append({
                                'ticker': ticker,
                                'allocation': 0.2,
                                'reason': 'Low risk, stable returns'
                            })
                        elif risk_level == "medium" and stock_risk in ["low", "medium"]:
                            allocation = 0.15 if stock_risk == "low" else 0.1
                            recommendations.append({
                                'ticker': ticker,
                                'allocation': allocation,
                                'reason': f'{stock_risk.title()} risk with good growth potential'
                            })
                        elif risk_level == "high":
                            allocation = 0.1 if stock_risk == "high" else 0.05
                            recommendations.append({
                                'ticker': ticker,
                                'allocation': allocation,
                                'reason': f'Diversified allocation for {risk_level} risk tolerance'
                            })
            
            return {
                'success': True,
                'risk_level': risk_level,
                'recommendations': recommendations,
                'total_allocation': sum(r['allocation'] for r in recommendations),
                'note': 'This is a simplified portfolio recommendation for educational purposes'
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Portfolio recommendation failed: {str(e)}"}

    def compare_stocks_function(self, tickers: List[str] = None) -> Dict[str, Any]:
        """Enhanced stock comparison function"""
        try:
            if not tickers:
                all_results = self.financial_service.analyze_all_stocks()
                stock_scores = []
                
                for ticker, result in all_results.items():
                    if result['success']:
                        stock_scores.append({
                            'ticker': ticker,
                            'company': result['data']['companyName'],
                            'score': result['data']['score'],
                            'valuation': result['data']['valuation']
                        })
                
                stock_scores.sort(key=lambda x: x['score'], reverse=True)
                
                # Add to memory
                self._add_to_memory("Compared all stocks", {
                    'action': 'stock_comparison',
                    'top_stock': stock_scores[0]['ticker'] if stock_scores else None
                }, importance=0.7)
                
                comparison_text = "📊 Stock Rankings by Analysis Score:\n"
                comparison_text += "-" * 50 + "\n"
                
                for i, stock in enumerate(stock_scores, 1):
                    comparison_text += f"{i:2d}. {stock['ticker']:5s} - {stock['company']:25s} | "
                    comparison_text += f"Score: {stock['score']:5.1f} | {stock['valuation']}\n"
                
                return {
                    'success': True,
                    'comparison': comparison_text,
                    'rankings': stock_scores,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                results = {}
                for ticker in tickers:
                    results[ticker] = self.financial_service.analyze_stock(ticker)
                
                return {'success': True, 'results': results}
                
        except Exception as e:
            return {'success': False, 'error': f"Comparison failed: {str(e)}"}

    def get_available_tickers_function(self) -> Dict[str, Any]:
        """Get available stock tickers"""
        try:
            tickers = self.financial_service.get_available_tickers()
            return {
                'success': True,
                'tickers': tickers,
                'message': f"Available tickers for analysis: {', '.join(tickers)}",
                'count': len(tickers)
            }
        except Exception as e:
            return {'success': False, 'error': f"Failed to get tickers: {str(e)}"}

    def _add_to_memory(self, content: str, context: Dict[str, Any], importance: float = 0.5):
        """Add information to memory system"""
        memory = Memory(
            content=content,
            timestamp=datetime.now(),
            context=context,
            importance=importance
        )
        self.conversation_memory.append(memory)
        
        # Keep memory manageable (last 50 items)
        if len(self.conversation_memory) > 50:
            self.conversation_memory = self.conversation_memory[-50:]

    def _get_relevant_memories(self, query: str, limit: int = 5) -> List[Memory]:
        """Retrieve relevant memories for context"""
        # Simple relevance scoring based on keywords
        query_words = set(query.lower().split())
        scored_memories = []
        
        for memory in self.conversation_memory:
            content_words = set(memory.content.lower().split())
            relevance = len(query_words.intersection(content_words)) * memory.importance
            if relevance > 0:
                scored_memories.append((memory, relevance))
        
        # Sort by relevance and return top results
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in scored_memories[:limit]]

    def _intelligent_tool_selection(self, user_message: str) -> List[Dict[str, Any]]:
        """Intelligently select tools based on user message"""
        message_lower = user_message.lower()
        selected_tools = []
        
        # Analyze message for different intents
        if any(word in message_lower for word in ['analyze', 'analysis', 'look at', 'check out']):
            # Extract ticker if mentioned
            for ticker in ['aapl', 'msft', 'googl', 'tsla', 'nvda', 'jpm', 'wmt']:
                if ticker in message_lower:
                    selected_tools.append({
                        'tool': 'analyze_stock',
                        'params': {'ticker': ticker.upper()},
                        'confidence': 0.9
                    })
                    break
        
        if any(word in message_lower for word in ['compare', 'rank', 'best', 'top', 'better']):
            selected_tools.append({
                'tool': 'compare_stocks',
                'params': {},
                'confidence': 0.8
            })
        
        if any(word in message_lower for word in ['risk', 'risky', 'safe', 'dangerous']):
            # Try to extract ticker for risk assessment
            for ticker in ['aapl', 'msft', 'googl', 'tsla', 'nvda', 'jpm', 'wmt']:
                if ticker in message_lower:
                    selected_tools.append({
                        'tool': 'risk_assessment',
                        'params': {'ticker': ticker.upper()},
                        'confidence': 0.7
                    })
                    break
        
        if any(word in message_lower for word in ['portfolio', 'invest', 'allocation', 'diversify']):
            # Try to extract risk level
            risk_level = "medium"  # default
            if any(word in message_lower for word in ['conservative', 'safe', 'low risk']):
                risk_level = "low"
            elif any(word in message_lower for word in ['aggressive', 'high risk', 'growth']):
                risk_level = "high"
            
            selected_tools.append({
                'tool': 'portfolio_recommendation',
                'params': {'risk_level': risk_level},
                'confidence': 0.6
            })
        
        if any(word in message_lower for word in ['available', 'list', 'tickers', 'stocks']):
            selected_tools.append({
                'tool': 'get_available_tickers',
                'params': {},
                'confidence': 0.9
            })
        
        return selected_tools

    async def _execute_reasoning_chain(self, user_message: str) -> Dict[str, Any]:
        """Execute multi-step reasoning process"""
        self.current_state = AgentState.THINKING
        
        reasoning_steps = [
            {"step": "understanding", "content": f"Understanding user request: {user_message}"},
            {"step": "memory_recall", "content": "Recalling relevant previous context"},
            {"step": "tool_selection", "content": "Selecting appropriate tools"},
            {"step": "execution", "content": "Executing selected tools"},
            {"step": "synthesis", "content": "Synthesizing results"},
            {"step": "reflection", "content": "Reflecting on response quality"}
        ]
        
        self.reasoning_chain = reasoning_steps
        
        # Get relevant memories
        relevant_memories = self._get_relevant_memories(user_message)
        
        # Select tools
        selected_tools = self._intelligent_tool_selection(user_message)
        
        # Execute tools
        self.current_state = AgentState.EXECUTING
        tool_results = []
        
        for tool_info in selected_tools:
            if tool_info['confidence'] >= self.available_tools[tool_info['tool']].confidence_threshold:
                tool_func = self.available_tools[tool_info['tool']].function
                result = tool_func(**tool_info['params'])
                tool_results.append({
                    'tool': tool_info['tool'],
                    'result': result,
                    'confidence': tool_info['confidence']
                })
        
        return {
            'memories': relevant_memories,
            'selected_tools': selected_tools,
            'tool_results': tool_results,
            'reasoning_chain': reasoning_steps
        }

    async def send_to_langbase(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Properly send messages to Langbase using the official SDK
        """
        try:
            if not self.langbase_client:
                self.logger.debug("Langbase client not initialized - check API key")
                return {"success": False, "message": "Langbase not configured"}
            
            # Prepare the data for Langbase
            pipe_data = {
                "message": message,
                "context": context or {},
                "timestamp": datetime.now().isoformat(),
                "session_id": getattr(self, 'session_id', 'default_session')
            }
            
            # Send to Langbase pipe
            response = await self.langbase_client.pipe.create(
                name=self.langbase_pipe_name,
                messages=[{"role": "user", "content": message}],
                variables=pipe_data
            )
            
            self.logger.info("Successfully sent message to Langbase")
            return {
                "success": True,
                "langbase_response": response,
                "message": "Message processed by Langbase"
            }
                
        except Exception as e:
            self.logger.warning(f"Langbase integration error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def chat_with_claude(self, user_message: str, execution_results: Optional[Dict] = None) -> str:
        """Enhanced Claude integration with reasoning context and conversation history"""
        try:
            messages = []
            
            # Add conversation history from memory
            for memory in self.conversation_memory:
                if 'user_message' in memory.context:
                    messages.append({
                        "role": "user",
                        "content": memory.context['user_message']
                    })
                    if 'assistant_response' in memory.context:
                        # Remove the "..." from the stored response
                        full_response = memory.context['assistant_response'].replace("...", "")
                        messages.append({
                            "role": "assistant",
                            "content": full_response
                        })
            
            # Add relevant memories as context
            if execution_results and execution_results.get('memories'):
                memory_context = "Previous relevant context:\n"
                for memory in execution_results['memories']:
                    memory_context += f"- {memory.content} ({memory.timestamp.strftime('%Y-%m-%d %H:%M')})\n"
                messages.append({
                    "role": "system",
                    "content": memory_context
                })
            
            # Add tool results if available
            if execution_results and execution_results.get('tool_results'):
                for tool_result in execution_results['tool_results']:
                    if tool_result['result']['success']:
                        messages.append({
                            "role": "assistant",
                            "content": f"Tool '{tool_result['tool']}' results: {json.dumps(tool_result['result'], indent=2)}"
                        })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })
            
            # Create completion with enhanced context
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                system=self.system_prompt,
                messages=messages
            )
            
            assistant_response = response.content[0].text
            
            # Add to memory
            self._add_to_memory(f"User asked: {user_message[:50]}...", {
                'user_message': user_message,
                'assistant_response': assistant_response,
                'tools_used': [tr['tool'] for tr in execution_results.get('tool_results', [])] if execution_results else []
            }, importance=0.6)
            
            return assistant_response
            
        except Exception as e:
            self.logger.error(f"Error in Claude API call: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    async def process_message(self, user_message: str) -> str:
        """
        Enhanced message processing with full agentic capabilities
        """
        try:
            self.logger.info(f"Processing message with agentic reasoning: {user_message[:100]}...")
            
            # Execute reasoning chain
            self.current_state = AgentState.PLANNING
            execution_results = await self._execute_reasoning_chain(user_message)
            
            # Get response from Claude with full context
            response = await self.chat_with_claude(user_message, execution_results)
            
            # Send to Langbase for analytics and logging
            langbase_result = await self.send_to_langbase(user_message, {
                'tools_executed': [tr['tool'] for tr in execution_results.get('tool_results', [])],
                'reasoning_steps': len(execution_results.get('reasoning_chain', [])),
                'memories_used': len(execution_results.get('memories', [])),
                'timestamp': datetime.now().isoformat()
            })
            
            # Reflection phase
            self.current_state = AgentState.REFLECTING
            self.logger.info("Completed agentic processing cycle")
            
            self.current_state = AgentState.LISTENING
            return response
            
        except Exception as e:
            self.logger.error(f"Error in agentic processing: {str(e)}")
            return f"I encountered an error while processing your message: {str(e)}"

    def start_enhanced_chat_session(self):
        """Start an enhanced interactive chat session"""
        print("🤖 Enhanced Agentic Financial Chatbot")
        print("=" * 60)
        print("I'm an advanced AI with reasoning, memory, and tool capabilities!")
        print(f"Available stocks: {', '.join(self.financial_service.get_available_tickers())}")
        print("\n🧠 My capabilities:")
        print("- Multi-step reasoning and planning")
        print("- Dynamic tool selection and usage")
        print("- Conversation memory and context")
        print("- Self-reflection and improvement")
        print("- Proper Langbase integration")
        print("\nType 'quit' or 'exit' to end the session.")
        print("=" * 60)
        
        while True:
            try:
                user_input = input(f"\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Thanks for using the Enhanced Agentic Chatbot! Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print(f"\n🧠 Thinking... (State: {self.current_state.value})")
                
                # Process message with full agentic capabilities
                response = asyncio.run(self.process_message(user_input))
                print(f"\n🤖 Assistant: {response}")
                
                # Show some internal state for transparency
                if len(self.conversation_memory) > 0:
                    print(f"\n💭 Memory items: {len(self.conversation_memory)} | Recent tools used")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session ended. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue

# Convenience functions
def create_enhanced_chatbot() -> EnhancedAgenticChatbot:
    """Create and return a new enhanced chatbot instance"""
    return EnhancedAgenticChatbot()

# Example usage
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        async def test_enhanced_chatbot():
            chatbot = create_enhanced_chatbot()
            
            test_messages = [
                "Hello! What can you do?",
                "Analyze Apple stock and assess its risk",
                "Compare all available stocks and recommend a low-risk portfolio",
                "I'm interested in tech stocks. What should I know about NVDA vs MSFT?"
            ]
            
            for message in test_messages:
                print(f"\n📝 Test: {message}")
                response = await chatbot.process_message(message)
                print(f"🤖 Response: {response}")
                print("-" * 80)
        
        asyncio.run(test_enhanced_chatbot())
    else:
        chatbot = create_enhanced_chatbot()
        chatbot.start_enhanced_chat_session()