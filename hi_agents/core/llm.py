import keyword
import os
from openai import OpenAI
from typing import Literal, Optional, Iterator
from .exception import MalAgentsException

SUPPORTED_PROVIDERS = Literal[
	"openai",
	"deepseek",
	"doubao",
	"qwen",
	"ollama",
	"custom"
]

class MalAgentsLLM:
	def __init__(
		self, 
		model: Optional[str] = None, 
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		provider: Optional[SUPPORTED_PROVIDERS] = None,
		temperature: float = 0.7,
		max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
		**kwargs,
	):
		'''
		初始化模型。优先使用传入的参数，如果没有传递，则从环境变量中获取

		Args:
			model: 模型ID，如果没有传递，则从 LLM_MODEL_ID 中获取
			api_key:  API密钥，如果没有传递，则从 LLM_API_KEY 中获取
			base_url: 基础URL，如果没有传递，则从 LLM_BASE_URL 中获取
		'''
		self.model = model or os.getenv("LLM_MODEL_ID")
		self.temperature = temperature
		self.max_tokens = max_tokens
		self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
		self.kwargs = kwargs

		# 如果没有提供 provider，尝试根据 model 名称推断
		if not provider and self.model:
			provider = self._infer_provider_from_model(self.model)
		
		request_provider = (provider or "").lower() if provider else None
		self.provider = request_provider
		# 自定义的直接从环境变量中取就可以
		if request_provider == 'custom':
			self.provider = "custom"
			self.api_key = os.getenv("LLM_API_KEY")
			self.base_url = os.getenv("LLM_BASE_URL")
		else:
			# 做一层适配
			self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

		# 模型层做适配
		if not self.model:
			self.model = self._get_default_model()
		if not all([self.api_key, self.base_url]):
			raise MalAgentsException('API密钥或基础URL未在 .env 文件中设置')

		# 创建 智能体 客户端
		self._client = self._create_client()

	def _infer_provider_from_model(self, model: str) -> Optional[str]:
		'''
		根据模型名称推断 provider
		'''
		model_lower = model.lower()
		if model_lower.startswith('doubao') or 'doubao' in model_lower:
			return 'doubao'
		elif model_lower.startswith('gpt') or 'openai' in model_lower:
			return 'openai'
		elif model_lower.startswith('deepseek') or 'deepseek' in model_lower:
			return 'deepseek'
		elif model_lower.startswith('qwen') or 'qwen' in model_lower:
			return 'qwen'
		elif 'ollama' in model_lower or 'llama' in model_lower:
			return 'ollama'
		return None

	def _resolve_credentials(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> tuple[str, str]:
		'''
		解析凭证和基础URL
		'''
		if self.provider == 'openai':
			resolve_api_key = api_key or os.getenv('OPENAI_API_KEY') or os.getenv("LLM_API_KEY")
			resolve_base_url = base_url or os.getenv('LLM_BASE_URL') or "https://api.openai.com/v1"
			if not resolve_api_key:
				raise ValueError('OPENAI_API_KEY is not set')
			if not resolve_base_url:
				raise ValueError('OPENAI_BASE_URL is not set')
			return resolve_api_key, resolve_base_url
		elif self.provider == 'deepseek':
			resolve_api_key = api_key or os.getenv('DEEPSEEK_API_KEY') or os.getenv("LLM_API_KEY")
			resolve_base_url = base_url or os.getenv('LLM_BASE_URL') or "https://api.deepseek.com"
			if not resolve_api_key:
				raise ValueError('DEEPSEEK_API_KEY is not set')
			if not resolve_base_url:
				raise ValueError('DEEPSEEK_BASE_URL is not set')
			return resolve_api_key, resolve_base_url
		elif self.provider == 'doubao':
			resolve_api_key = api_key or os.getenv('DOUBAO_API_KEY') or os.getenv("LLM_API_KEY")
			resolve_base_url = base_url or os.getenv('LLM_BASE_URL') or "https://ark.cn-beijing.volces.com/api/v3"
			if not resolve_api_key:
				raise ValueError('DOUBAO_API_KEY is not set')
			if not resolve_base_url:
				raise ValueError('DOUBAO_BASE_URL is not set')
			return resolve_api_key, resolve_base_url
		elif self.provider == 'qwen':
			resolve_api_key = api_key or os.getenv('QWEN_API_KEY') or os.getenv("LLM_API_KEY")
			resolve_base_url = base_url or os.getenv('LLM_BASE_URL') or "https://dashscope.aliyuncs.com/compatible-mode/v1"
			if not resolve_api_key:
				raise ValueError('QWEN_API_KEY is not set')
			if not resolve_base_url:
				raise ValueError('QWEN_BASE_URL is not set')
			return resolve_api_key, resolve_base_url
		elif self.provider == 'ollama':
			resolve_api_key = api_key or os.getenv('OLLAMA_API_KEY') or os.getenv("LLM_API_KEY")
			resolve_base_url = base_url or os.getenv('LLM_BASE_URL') or "http://localhost:11434/v1"
			if not resolve_api_key:
				raise ValueError('OLLAMA_API_KEY is not set')
			if not resolve_base_url:
				raise ValueError('OLLAMA_BASE_URL is not set')
			return resolve_api_key, resolve_base_url
		else:
			resolve_api_key = api_key or os.getenv('LLM_API_KEY')
			resolve_base_url = base_url or os.getenv('LLM_BASE_URL')
			if not resolve_api_key:
				raise ValueError('LLM_API_KEY is not set')
			if not resolve_base_url:
				raise ValueError('LLM_BASE_URL is not set')
			return resolve_api_key, resolve_base_url

	def _get_default_model(self) -> str:
		'''
		获取默认模型
		'''
		if self.provider == 'openai':
			return "gpt-3.5-turbo"
		elif self.provider == 'deepseek':
			return "deepseek-chat"
		elif self.provider == 'doubao':
			return "doubao-seed-1-8-251228"
		elif self.provider == 'qwen':
			return "qwen-plus"
		elif self.provider == 'ollama':
			return "llama3.1:8b"
		else:
			return "gpt-3.5-turbo"

	def _create_client(self) -> OpenAI:
		'''
		创建OpenAI客户端
		'''
		return OpenAI(
			api_key=self.api_key, 
			base_url=self.base_url, 
			timeout=self.timeout
		)
	def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
		'''
		调用大语言模型进行思考，并返回流式响应。
		默认获取流式数据，可以设置temperature参数来控制思考的随机性。

		Args:
			messages: 对话历史，列表类型，每个元素是一个字典，包含role和content两个键。
			temperature: 思考的随机性，取值范围为0到1，默认值为0.7。

		Returns:
			Generator[str, None, None]: 流式响应生成器。
		'''
		print(f'🧠 正在调用 {self.model} 模型...')
		try:
			response = self._client.chat.completions.create(
				model=self.model,
				messages = messages,
				temperature = temperature if temperature is not None else self.temperature,
				max_tokens = self.max_tokens,
				timeout = self.timeout,
				stream = True,
			)
			print('✅ 调用LLM API成功')
			# 处理流式数据
			for chunk in response:
				content = chunk.choices[0].delta.content or ""
				if content:
					print(content, end="", flush=True)
					# 创建生成器返回给调用者「也就是将 content 发送给 think 的调用者」
					# 当调用者请求下一个值时，从这里继续执行
					yield content 
					'''
					# 想象这是调用think函数的过程
					generator = agent.think(messages)  # 1. 创建生成器对象
					# 此时函数还没真正执行！

					# 2. 开始迭代
					first_chunk = next(generator)  
					# ↑ think函数开始执行，遇到第一个yield时暂停
					# 返回第一个content给first_chunk

					second_chunk = next(generator)
					# ↑ 从暂停处继续执行，遇到下一个yield时暂停
					# 返回第二个content给second_chunk

					# ... 直到for循环结束
					'''
			print() # 换行
		except Exception as e:
			print(f"❌ 调用LLM API时发生错误: {e}")
			raise MalAgentsException(f"调用LLM API时发生错误: {e}")
	
	def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
		'''
		非流式数据返回
		'''
		print(f'💬 正在调用 {self.model} 模型...')
		try:
			response = self._client.chat.completions.create(
				model=self.model,
				messages = messages,
				temperature = kwargs.get('temperature', self.temperature),
				max_tokens = kwargs.get('max_tokens', self.max_tokens),
				**{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
			)
			return response.choices[0].message.content
		except Exception as e:
			print(f"❌ 调用LLM API时发生错误: {e}")
			raise MalAgentsException(f"调用LLM API时发生错误: {e}")
	def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
		'''
		流式调用LLM的别名方法，与think方法功能相同。
		保持向后兼容性。
		'''
		temperature = kwargs.get('temperature')
		yield from self.think(messages, temperature)

