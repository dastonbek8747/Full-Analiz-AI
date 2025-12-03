import streamlit as st
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import tempfile
import time
from typing import List, Dict, Optional
import requests
from io import BytesIO
from dataclasses import dataclass
import json

# Safe imports with error handling
try:
    from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.error("⚠️ transformers kutubxonasi o'rnatilmagan!")

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ opencv-python o'rnatilmagan. Video tahlili ishlamaydi!")


@dataclass
class AnalysisResult:
    """Tahlil natijasi"""
    success: bool
    description: str
    objects_detected: List[str]
    details: Dict[str, str]
    confidence: float
    time_taken: float
    web_search_results: List[Dict] = None
    error: str = ""


class LLaVAVideoAnalyzer:
    """
    LLaVA-NeXT Video model bilan rasm va videolarni
    batafsil tahlil qiluvchi tizim + Web Search
    """

    def __init__(self, device: str = "auto"):
        self.device = device
        self.processor = None
        self.model = None
        self.google_api_key = ""
        self.google_cx = ""
        self._load_api_keys()

    def _load_api_keys(self):
        """API kalitlarini yuklash"""
        try:
            self.google_api_key = st.secrets.get("GOOGLE_API_KEY", "")
            self.google_cx = st.secrets.get("GOOGLE_CX", "")
        except Exception:
            pass

    def load_model(self):
        """Modelni yuklash (cache qilingan)"""
        if not TRANSFORMERS_AVAILABLE:
            st.error("❌ transformers kutubxonasi o'rnatilmagan!")
            return False

        try:
            with st.spinner("🤖 Model yuklanmoqda... (Bu 5-10 daqiqa olishi mumkin)"):
                self.processor = LlavaNextVideoProcessor.from_pretrained(
                    "llava-hf/LLaVA-NeXT-Video-7B-hf"
                )

                self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
                    "llava-hf/LLaVA-NeXT-Video-7B-hf",
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    low_cpu_mem_usage=True
                )

                st.success("✅ Model muvaffaqiyatli yuklandi!")
                return True
        except Exception as e:
            st.error(f"❌ Model yuklashda xatolik: {str(e)}")
            return False

    def analyze_image(
            self,
            image: Image.Image,
            custom_prompt: Optional[str] = None
    ) -> AnalysisResult:
        """
        Rasmni batafsil tahlil qilish
        """
        start_time = time.time()

        if self.model is None:
            if not self.load_model():
                return AnalysisResult(
                    success=False,
                    description="",
                    objects_detected=[],
                    details={},
                    confidence=0.0,
                    time_taken=0.0,
                    error="Model yuklanmadi"
                )

        try:
            # RGB formatga o'tkazish
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Batafsil tahlil uchun maxsus prompt
            if custom_prompt is None:
                custom_prompt = """Analyze this image in extreme detail and provide:

1. MAIN OBJECTS: What are the primary subjects? (e.g., "black Lamborghini car", "white Pitbull dog")
2. SPECIFIC DETAILS:
   - For vehicles: Brand, model, color, type (sports car, SUV, etc.)
   - For animals: Breed, color, size, activity
   - For people: Number, clothing, actions
   - For objects: Type, color, material, purpose
3. COLORS: Dominant and secondary colors
4. LOCATION & SETTING: Where is this? Indoor/outdoor? Background details
5. ACTION & CONTEXT: What's happening in the scene?
6. QUALITY & STYLE: Photo quality, lighting, composition

Be VERY specific with names, brands, models, and breeds. 
Example: "This is a matte black Lamborghini Aventador parked on a city street"
NOT just "a car"."""

            # Conversation yaratish
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": custom_prompt},
                        {"type": "image", "image": image},
                    ],
                }
            ]

            # Prompt tayyorlash
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )

            # Input tayyorlash
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.model.device)

            # Generate qilish
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            # Decode qilish
            description = self.processor.batch_decode(
                output,
                skip_special_tokens=True
            )[0]

            # Javobni tozalash (assistant: prefixni olib tashlash)
            if "assistant" in description.lower():
                description = description.split("assistant", 1)[-1].strip()
                description = description.lstrip(":").strip()

            # Obyektlarni ajratib olish
            objects = self._extract_objects(description)
            details = self._extract_details(description)

            elapsed_time = time.time() - start_time

            return AnalysisResult(
                success=True,
                description=description,
                objects_detected=objects,
                details=details,
                confidence=0.9,
                time_taken=elapsed_time
            )

        except Exception as e:
            return AnalysisResult(
                success=False,
                description="",
                objects_detected=[],
                details={},
                confidence=0.0,
                time_taken=time.time() - start_time,
                error=str(e)
            )

    def analyze_video(
            self,
            video_path: str,
            num_frames: int = 8
    ) -> AnalysisResult:
        """
        Videoni tahlil qilish
        """
        if not CV2_AVAILABLE:
            return AnalysisResult(
                success=False,
                description="",
                objects_detected=[],
                details={},
                confidence=0.0,
                time_taken=0.0,
                error="opencv-python o'rnatilmagan! Video tahlili mumkin emas."
            )

        start_time = time.time()

        if self.model is None:
            if not self.load_model():
                return AnalysisResult(
                    success=False,
                    description="",
                    objects_detected=[],
                    details={},
                    confidence=0.0,
                    time_taken=0.0,
                    error="Model yuklanmadi"
                )

        try:
            # Video'dan framelar olish
            frames = self._extract_video_frames(video_path, num_frames)

            if not frames:
                return AnalysisResult(
                    success=False,
                    description="",
                    objects_detected=[],
                    details={},
                    confidence=0.0,
                    time_taken=0.0,
                    error="Video framelar olinmadi"
                )

            # Video tahlil prompti
            prompt_text = """Analyze this video and describe:

1. MAIN CONTENT: What is shown in this video?
2. OBJECTS/SUBJECTS: Identify specific objects, people, animals with details (brand, model, breed, etc.)
3. ACTIONS: What activities or movements are happening?
4. COLORS & APPEARANCE: Dominant colors and visual characteristics
5. SETTING: Location and environment
6. SEQUENCE: How does the content change throughout the video?

Be specific with identifications!"""

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "video", "video": frames},
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=prompt,
                videos=frames,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            description = self.processor.batch_decode(
                output,
                skip_special_tokens=True
            )[0]

            # Javobni tozalash
            if "assistant" in description.lower():
                description = description.split("assistant", 1)[-1].strip()
                description = description.lstrip(":").strip()

            objects = self._extract_objects(description)
            details = self._extract_details(description)

            elapsed_time = time.time() - start_time

            return AnalysisResult(
                success=True,
                description=description,
                objects_detected=objects,
                details=details,
                confidence=0.85,
                time_taken=elapsed_time
            )

        except Exception as e:
            return AnalysisResult(
                success=False,
                description="",
                objects_detected=[],
                details={},
                confidence=0.0,
                time_taken=time.time() - start_time,
                error=str(e)
            )

    def _extract_video_frames(
            self,
            video_path: str,
            num_frames: int
    ) -> Optional[List[Image.Image]]:
        """Video'dan framelar ajratib olish"""
        if not CV2_AVAILABLE:
            return None

        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames == 0:
                return None

            # Bir tekis taqsimlangan frame'lar
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if ret:
                    # BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)

            cap.release()
            return frames if frames else None

        except Exception as e:
            st.error(f"Frame extraction xatosi: {str(e)}")
            return None

    def _extract_objects(self, description: str) -> List[str]:
        """Tavsifdan obyektlarni ajratib olish"""
        objects = []

        # Oddiy keyword extraction
        keywords = [
            "car", "mashina", "vehicle", "avtomobil",
            "dog", "it", "cat", "mushuk",
            "person", "odam", "people",
            "building", "bino", "house", "uy",
            "tree", "daraxt", "animal", "hayvon"
        ]

        desc_lower = description.lower()
        for keyword in keywords:
            if keyword in desc_lower:
                # Context bilan olish
                start = desc_lower.find(keyword)
                context = description[max(0, start - 20):min(len(description), start + 50)]
                objects.append(context.strip())

        return list(set(objects))[:10]  # Unique va max 10 ta

    def _extract_details(self, description: str) -> Dict[str, str]:
        """Detallarni ajratib olish"""
        details = {}

        lines = description.split('\n')
        current_key = ""

        for line in lines:
            if ':' in line and len(line.split(':')[0]) < 30:
                key, value = line.split(':', 1)
                current_key = key.strip()
                details[current_key] = value.strip()
            elif current_key and line.strip():
                details[current_key] += " " + line.strip()

        return details

    def web_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Google Search API orqali qidirish"""
        if not self.google_api_key or not self.google_cx:
            return []

        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cx,
                "q": query,
                "searchType": "image",
                "num": num_results,
                "safe": "active"
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])

                results = []
                for item in items:
                    results.append({
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'thumbnail': item.get('image', {}).get('thumbnailLink', ''),
                        'snippet': item.get('snippet', ''),
                        'context': item.get('displayLink', '')
                    })

                return results

            return []

        except Exception as e:
            st.warning(f"Web search xatosi: {str(e)}")
            return []

    def analyze_with_search(
            self,
            image: Image.Image
    ) -> Dict:
        """Rasm tahlili + Web qidiruv"""

        # 1. Rasmni tahlil qilish
        analysis = self.analyze_image(image)

        if not analysis.success:
            return {
                'analysis': analysis,
                'search_results': [],
                'search_query': ''
            }

        # 2. Qidiruv so'rovini yaratish
        search_query = self._generate_search_query(analysis)

        # 3. Web qidiruv
        search_results = []
        if search_query and self.google_api_key:
            search_results = self.web_search(search_query)

        return {
            'analysis': analysis,
            'search_results': search_results,
            'search_query': search_query
        }

    def _generate_search_query(self, analysis: AnalysisResult) -> str:
        """Tahlil asosida aqlli qidiruv so'rovi yaratish"""

        # Obyektlardan eng muhimini olish
        if analysis.objects_detected:
            # Eng birinchi va eng uzun obyekt (ko'proq detallar)
            main_object = max(analysis.objects_detected, key=len)
            return main_object[:100]  # Max 100 belgi

        # Agar obyekt yo'q bo'lsa, tavsifdan birinchi 50 belgi
        words = analysis.description.split()[:8]
        return ' '.join(words)


def main():
    """Asosiy Streamlit dastur"""

    st.set_page_config(
        page_title="🎬 LLaVA Video/Image Analyzer",
        page_icon="🎬",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .big-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .result-container {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="big-title">🎬 LLaVA Video/Image Analyzer</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-box">
    <h3>🚀 Kuchli AI Vision + Web Search</h3>
    Rasmlar va videolarni batafsil tahlil qiling va kerak bo'lsa internetdan qo'shimcha ma'lumot toping!

    <b>Imkoniyatlar:</b>
    • 🖼️ Rasm tahlili (batafsil identifikatsiya)
    • 🎥 Video tahlili (harakat va kontekst)
    • 🔍 Google Search integratsiyasi
    • 🎯 Aniq obyekt tanish (marka, model, tur)
    • 🎨 Rang va detallarni aniqlash
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Sozlamalar")

        # Analyzer yaratish
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = LLaVAVideoAnalyzer()

        analyzer = st.session_state.analyzer

        # Model holati
        st.markdown("---")
        st.subheader("🤖 Model Holati")

        if st.button("📥 Modelni Yuklash", use_container_width=True):
            analyzer.load_model()

        # Google Search
        st.markdown("---")
        st.subheader("🔍 Web Search")

        use_search = st.checkbox("Google Search", value=False)

        if use_search and not analyzer.google_api_key:
            st.warning("⚠️ API keys yo'q!")
            st.info("""
            `.streamlit/secrets.toml`:
            ```toml
            GOOGLE_API_KEY = "key"
            GOOGLE_CX = "cx"
            ```
            """)

        # System info
        st.markdown("---")
        st.subheader("💻 System Info")

        device = "CUDA" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device)

        if torch.cuda.is_available():
            st.metric("GPU", torch.cuda.get_device_name(0))
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.metric("VRAM", f"{memory:.1f} GB")

        # Dependencies holati
        st.markdown("---")
        st.subheader("📦 Dependencies")
        st.metric("Transformers", "✅" if TRANSFORMERS_AVAILABLE else "❌")
        st.metric("OpenCV", "✅" if CV2_AVAILABLE else "❌")

    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "📸 Rasm Tahlili",
        "🎥 Video Tahlili",
        "📚 Yo'riqnoma"
    ])

    with tab1:
        st.header("📸 Rasm Tahlili")

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("📤 Rasm Yuklash")

            uploaded_file = st.file_uploader(
                "Rasmni tanlang",
                type=['jpg', 'jpeg', 'png', 'webp', 'bmp']
            )

            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

                st.markdown("**📊 Rasm Ma'lumotlari**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Kenglik", f"{image.size[0]}px")
                with col_b:
                    st.metric("Balandlik", f"{image.size[1]}px")

        with col2:
            if uploaded_file:
                st.subheader("🤖 AI Tahlil")

                # Custom prompt
                custom_prompt = st.text_area(
                    "Maxsus savol (ixtiyoriy)",
                    placeholder="Masalan: Bu qanday mashina?",
                    height=80
                )

                # Analysis options
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    if st.button("🔍 Oddiy Tahlil", use_container_width=True, type="primary"):
                        with st.spinner("Tahlil qilinmoqda..."):
                            result = analyzer.analyze_image(
                                image,
                                custom_prompt if custom_prompt.strip() else None
                            )

                        if result.success:
                            st.success(f"✅ Tayyor! ({result.time_taken:.1f}s)")

                            st.markdown('<div class="result-container">', unsafe_allow_html=True)
                            st.markdown("### 📝 Tahlil Natijasi")
                            st.write(result.description)

                            if result.objects_detected:
                                st.markdown("---")
                                st.markdown("**🎯 Topilgan Obyektlar:**")
                                for obj in result.objects_detected[:5]:
                                    st.info(obj)

                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error(f"❌ {result.error}")

                with col_opt2:
                    if st.button("🌐 Tahlil + Search", use_container_width=True):
                        if not use_search:
                            st.warning("⚠️ Web search yoqilmagan!")
                        else:
                            with st.spinner("Tahlil va qidiruv..."):
                                result_data = analyzer.analyze_with_search(image)

                            result = result_data['analysis']

                            if result.success:
                                st.success(f"✅ Tayyor! ({result.time_taken:.1f}s)")

                                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                                st.markdown("### 📝 Tahlil")
                                st.write(result.description)
                                st.markdown('</div>', unsafe_allow_html=True)

                                # Search results
                                if result_data['search_results']:
                                    st.markdown("### 🔍 O'xshash Rasmlar")
                                    st.caption(f"Qidiruv: {result_data['search_query']}")

                                    cols = st.columns(3)
                                    for idx, res in enumerate(result_data['search_results'][:3]):
                                        with cols[idx]:
                                            st.image(res['thumbnail'], use_container_width=True)
                                            st.caption(res['title'][:40])
                                            st.link_button("Ko'rish", res['link'], use_container_width=True)

    with tab2:
        st.header("🎥 Video Tahlili")

        if not CV2_AVAILABLE:
            st.error("❌ opencv-python o'rnatilmagan! Video tahlili ishlamaydi.")
            st.info(
                "Streamlit Cloud uchun `requirements.txt` faylida quyidagilarni qo'shing:\n```\nlibgl1-mesa-glx\nlibglib2.0-0\n```")
        else:
            st.info("🎬 Video tahlili uchun video fayl yuklang (MP4, AVI, MOV)")

            video_file = st.file_uploader(
                "Video tanlang",
                type=['mp4', 'avi', 'mov', 'mkv']
            )

            if video_file:
                # Temporary faylga saqlash
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(video_file.read())
                    video_path = tmp_file.name

                col1, col2 = st.columns([1, 1.2])

                with col1:
                    st.video(video_path)

                    num_frames = st.slider("Frame soni", 4, 16, 8)

                with col2:
                    if st.button("🎬 Videoni Tahlil Qilish", type="primary", use_container_width=True):
                        with st.spinner(f"Video tahlil qilinmoqda ({num_frames} frame)..."):
                            result = analyzer.analyze_video(video_path, num_frames)

                        if result.success:
                            st.success(f"✅ Tayyor! ({result.time_taken:.1f}s)")

                            st.markdown('<div class="result-container">', unsafe_allow_html=True)
                            st.markdown("### 🎬 Video Tahlili")
                            st.write(result.description)

                            if result.objects_detected:
                                st.markdown("---")
                                st.markdown("**🎯 Videoda Topilganlar:**")
                                for obj in result.objects_detected[:5]:
                                    st.info(obj)

                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error(f"❌ {result.error}")

    with tab3:
        st.header("📚 To'liq Yo'riqnoma")

        st.markdown("""
        ## 🚀 O'rnatish

        ### 1️⃣ Dependencies

        **requirements.txt:**
        ```
        streamlit
        transformers
        torch
        pillow
        opencv-python-headless
        requests
        numpy
        accelerate
        ```

        ### 2️⃣ Streamlit Cloud uchun

        **requirements.txt:**
        ```
        libgl1-mesa-glx
        libglib2.0-0
        ```

        ### 3️⃣ Model Yuklanishi
        Birinchi ishga tushirishda ~14GB model yuklanadi (bir marta).

        ### 4️⃣ Hardware Talablar

        | Komponent | Minimum | Tavsiya |
        |-----------|---------|---------|
        | RAM | 16 GB | 32 GB |
        | GPU VRAM | 8 GB | 16 GB+ |
        | Disk | 20 GB | 50 GB |

        ---

        ## 💡 Foydalanish

        ### Rasm Tahlili:
        1. Rasm yuklang (JPG, PNG va h.k.)
        2. "Oddiy Tahlil" yoki "Tahlil + Search" ni tanlang
        3. Maxsus savol bering (ixtiyoriy)
        4. Natijani ko'ring!

        ### Video Tahlili:
        1. Video yuklang (MP4, AVI va h.k.)
        2. Frame sonini tanlang (4-16)
        3. "Videoni Tahlil Qilish" bosing
        4. Batafsil tahlilni o'qing!

        ---

        ## 🔍 Google Search Sozlash

        `.streamlit/secrets.toml` yarating:
        ```toml
        GOOGLE_API_KEY = "AIzaSy..."
        GOOGLE_CX = "017576..."
        ```

        ---

        ## 🎯 Misol Natijalar

        **Input:** Qora sport mashina rasmi

        **Output:**
        ```
        This is a matte black Lamborghini Aventador SVJ, 
        a high-performance sports car. The vehicle features:
        - Color: Matte black finish
        - Model: Aventador SVJ (Super Veloce Jota)
        - Body type: Mid-engine supercar
        - Setting: Parked on an urban street
        - Notable features: Aggressive aerodynamic design, 
          large rear wing, angular bodywork
        ```

        ---

        ## ⚠️ Muammolar

        **ImportError: cv2**
        - `opencv-python-headless` o'rnating (Streamlit Cloud uchun)
        - `requirements.txt` faylida `libgl1-mesa-glx` qo'shing

        **CUDA xatosi:**
        ```bash
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ```

        **Xotira yetmayapti:**
        - Rasmni kichiklashtiring
        - `torch_dtype=torch.float16` ishlatilganini tekshiring
        - GPU ishlatayotganingizni tekshiring

        **Model sekin:**
        - GPU ishlatilganini tekshiring: `torch.cuda.is_available()`
        - Batch size'ni kamaytiring
        - 4-bit quantization: `load_in_4bit=True`
        """)


if __name__ == "__main__":
    main()