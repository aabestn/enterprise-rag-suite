import fitz  # PyMuPDF
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class VisionExtractor:
    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def extract_page_images(self, pdf_bytes: bytes) -> list[dict]:
        """
        Extracts high-resolution images of each page from multi-page PDFs
        for visual analysis and vision-LLM processing.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        extracted_pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=self.dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            # Extract plain text fallback alongside layout vision frame
            text_content = page.get_text("text")

            extracted_pages.append({
                "page_number": page_num + 1,
                "image": img,
                "raw_text": text_content,
                "width": pix.width,
                "height": pix.height
            })
            
        logger.info(f"Successfully processed {len(extracted_pages)} pages via vision extractor.")
        return extracted_pages