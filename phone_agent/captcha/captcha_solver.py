"""
CAPTCHA Solver Module - Automatic CAPTCHA recognition with multiple strategies.

Supports:
- VLM-based recognition (using AutoGLM model)
- Local OCR fallback (using ddddocr)
- Multiple CAPTCHA types: text, slider, click
"""

import base64
import io
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CaptchaType(Enum):
    """Supported CAPTCHA types."""
    TEXT = "text"        # Text/number CAPTCHA
    SLIDER = "slider"    # Slider CAPTCHA
    CLICK = "click"      # Click-based CAPTCHA (e.g., "click all cars")


@dataclass
class CaptchaResult:
    """Result of CAPTCHA solving attempt."""
    success: bool
    value: str | list[tuple[int, int]] | int | None  # str for text, list for click positions, int for slider distance
    confidence: float = 0.0
    method: str = ""  # "vlm" or "ocr"
    error: str | None = None


class CaptchaSolver:
    """
    CAPTCHA solver with multiple recognition strategies.
    
    Uses a hybrid approach:
    1. First attempts VLM-based recognition using the AutoGLM model
    2. Falls back to local OCR (ddddocr) if VLM fails
    
    Args:
        model_client: ModelClient instance for VLM-based recognition
        use_ocr: Whether to enable local OCR fallback
        max_retries: Maximum number of solving attempts
    """
    
    def __init__(
        self,
        model_client=None,
        use_ocr: bool = True,
        max_retries: int = 3,
    ):
        self.model_client = model_client
        self.use_ocr = use_ocr
        self.max_retries = max_retries
        self._ocr = None  # Lazy-loaded OCR instance
    
    def solve(
        self,
        screenshot_base64: str,
        captcha_type: str = "text",
        hint: str | None = None,
    ) -> CaptchaResult:
        """
        Attempt to solve a CAPTCHA.
        
        Args:
            screenshot_base64: Base64-encoded screenshot of the CAPTCHA
            captcha_type: Type of CAPTCHA ("text", "slider", "click")
            hint: Optional hint about the CAPTCHA (e.g., "输入图中的数字")
        
        Returns:
            CaptchaResult with solving result
        """
        captcha_enum = CaptchaType(captcha_type) if isinstance(captcha_type, str) else captcha_type
        
        # Strategy 1: Try VLM-based recognition
        if self.model_client:
            result = self._solve_with_vlm(screenshot_base64, captcha_enum, hint)
            if result.success:
                logger.info(f"CAPTCHA solved with VLM: {result.value}")
                return result
            logger.warning(f"VLM failed to solve CAPTCHA: {result.error}")
        
        # Strategy 2: Fall back to local OCR for text CAPTCHAs
        if self.use_ocr and captcha_enum == CaptchaType.TEXT:
            result = self._solve_with_ocr(screenshot_base64)
            if result.success:
                logger.info(f"CAPTCHA solved with OCR: {result.value}")
                return result
            logger.warning(f"OCR failed to solve CAPTCHA: {result.error}")
        
        return CaptchaResult(
            success=False,
            value=None,
            error="All solving strategies failed"
        )
    
    def _solve_with_vlm(
        self,
        screenshot_base64: str,
        captcha_type: CaptchaType,
        hint: str | None = None,
    ) -> CaptchaResult:
        """Use VLM model to solve CAPTCHA."""
        if not self.model_client:
            return CaptchaResult(success=False, value=None, error="No model client")
        
        try:
            # Build prompt based on CAPTCHA type
            if captcha_type == CaptchaType.TEXT:
                prompt = self._build_text_captcha_prompt(hint)
            elif captcha_type == CaptchaType.SLIDER:
                prompt = self._build_slider_captcha_prompt(hint)
            elif captcha_type == CaptchaType.CLICK:
                prompt = self._build_click_captcha_prompt(hint)
            else:
                return CaptchaResult(success=False, value=None, error=f"Unsupported type: {captcha_type}")
            
            # Build messages for model
            messages = [
                {"role": "system", "content": "你是一个验证码识别专家。请仔细观察图片中的验证码并准确识别。"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Get model response
            response = self.model_client.request(messages)
            
            # Parse response based on CAPTCHA type
            return self._parse_vlm_response(response.raw_content, captcha_type)
            
        except Exception as e:
            logger.exception("VLM CAPTCHA solving failed")
            return CaptchaResult(success=False, value=None, error=str(e))
    
    def _solve_with_ocr(self, screenshot_base64: str) -> CaptchaResult:
        """Use local OCR to solve text CAPTCHA."""
        try:
            # Lazy load ddddocr
            if self._ocr is None:
                try:
                    import ddddocr
                    self._ocr = ddddocr.DdddOcr(show_ad=False)
                except ImportError:
                    return CaptchaResult(
                        success=False,
                        value=None,
                        error="ddddocr not installed. Run: pip install ddddocr"
                    )
            
            # Decode image and recognize
            image_bytes = base64.b64decode(screenshot_base64)
            result = self._ocr.classification(image_bytes)
            
            if result and len(result) > 0:
                return CaptchaResult(
                    success=True,
                    value=result,
                    confidence=0.8,  # ddddocr doesn't provide confidence
                    method="ocr"
                )
            
            return CaptchaResult(success=False, value=None, error="OCR returned empty result")
            
        except Exception as e:
            logger.exception("OCR CAPTCHA solving failed")
            return CaptchaResult(success=False, value=None, error=str(e))
    
    def _build_text_captcha_prompt(self, hint: str | None = None) -> str:
        """Build prompt for text CAPTCHA recognition."""
        base_prompt = """请识别图片中的验证码。

要求：
1. 仔细观察验证码区域的文字或数字
2. 只输出验证码内容，不要输出其他任何文字
3. 如果有多个可能的结果，选择最可能的一个

请直接输出验证码内容："""
        
        if hint:
            base_prompt = f"提示：{hint}\n\n" + base_prompt
        
        return base_prompt
    
    def _build_slider_captcha_prompt(self, hint: str | None = None) -> str:
        """Build prompt for slider CAPTCHA recognition."""
        return """请分析图片中的滑块验证码。

要求：
1. 找到滑块需要移动到的目标位置
2. 计算滑块需要移动的水平距离（像素）
3. 输出格式：SLIDER_DISTANCE:数字

例如：SLIDER_DISTANCE:150

请输出滑动距离："""
    
    def _build_click_captcha_prompt(self, hint: str | None = None) -> str:
        """Build prompt for click CAPTCHA recognition."""
        base_prompt = """请分析图片中的点选验证码。

要求：
1. 找到需要点击的所有目标位置
2. 输出每个目标的坐标（相对于图片，0-1000范围）
3. 输出格式：CLICK_POINTS:[[x1,y1],[x2,y2],...]

例如：CLICK_POINTS:[[100,200],[500,300],[800,400]]

请输出点击坐标："""
        
        if hint:
            base_prompt = f"提示：{hint}\n\n" + base_prompt
        
        return base_prompt
    
    def _parse_vlm_response(self, content: str, captcha_type: CaptchaType) -> CaptchaResult:
        """Parse VLM response based on CAPTCHA type."""
        content = content.strip()
        
        try:
            if captcha_type == CaptchaType.TEXT:
                # Extract text content, clean up any extra text
                # Look for alphanumeric content
                import re
                # Remove common wrapper text
                content = content.replace("验证码是", "").replace("验证码：", "").replace("验证码:", "")
                content = content.replace("答案是", "").replace("答案：", "").replace("答案:", "")
                # Extract alphanumeric characters
                match = re.search(r'[a-zA-Z0-9]{4,8}', content)
                if match:
                    return CaptchaResult(success=True, value=match.group(), confidence=0.9, method="vlm")
                # If no match, try to clean and return
                cleaned = re.sub(r'[^a-zA-Z0-9]', '', content)
                if cleaned:
                    return CaptchaResult(success=True, value=cleaned, confidence=0.7, method="vlm")
                return CaptchaResult(success=False, value=None, error="Could not extract text CAPTCHA")
            
            elif captcha_type == CaptchaType.SLIDER:
                # Extract slider distance
                import re
                match = re.search(r'SLIDER_DISTANCE[:\s]*(\d+)', content)
                if match:
                    distance = int(match.group(1))
                    return CaptchaResult(success=True, value=distance, confidence=0.8, method="vlm")
                # Try to find any number
                match = re.search(r'(\d+)', content)
                if match:
                    distance = int(match.group(1))
                    return CaptchaResult(success=True, value=distance, confidence=0.6, method="vlm")
                return CaptchaResult(success=False, value=None, error="Could not extract slider distance")
            
            elif captcha_type == CaptchaType.CLICK:
                # Extract click coordinates
                import re
                import ast
                match = re.search(r'CLICK_POINTS[:\s]*(\[\[.*?\]\])', content)
                if match:
                    points = ast.literal_eval(match.group(1))
                    return CaptchaResult(success=True, value=points, confidence=0.8, method="vlm")
                return CaptchaResult(success=False, value=None, error="Could not extract click points")
            
        except Exception as e:
            return CaptchaResult(success=False, value=None, error=f"Parse error: {e}")
        
        return CaptchaResult(success=False, value=None, error="Unknown CAPTCHA type")
