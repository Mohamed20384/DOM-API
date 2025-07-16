import json
import os
import argparse
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx2pdf import convert


def process_restaurants_json(input_file_path: str):
    """Process JSON file and generate PDFs for eShop restaurants. Also save partial info for others."""

    os.makedirs("Restaurants_Word", exist_ok=True)
    os.makedirs("Restaurants_PDF", exist_ok=True)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    no_eshop_info = []

    for rest in data:
        name = rest.get("name", "مطعم بدون اسم")
        address = rest.get("adress", "")
        phone = rest.get("phone", [])
        rank = rest.get("rank", "غير مصنف")
        bestsell = rest.get("bestsell", "غير محدد")
        e_shop = rest.get("eShop", False)

        if not e_shop:
            no_eshop_info.append({
                "name": name,
                "address": address,
                "phone": phone,
                "bestsell": bestsell
            })
            continue

        real_rates = rest.get("realRates", {})
        emenu = rest.get("eMenu", {})
        open_time = rest.get("openTime", "غير معروف")
        close_time = rest.get("closeTime", "غير معروف")
        cooking_time = rest.get("cookingTimeRange", "غير محدد")

        doc = Document()

        def write_paragraph(label, value):
            if isinstance(value, list):
                value = "، ".join(value)
            elif isinstance(value, dict):
                value = "\n".join([f"{k}: {v}" for k, v in value.items()])
            text = f"{label}: {value}"
            para = doc.add_paragraph(text)
            para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            run = para.runs[0]
            run.font.name = 'Calibri'
            run.font.size = Pt(14)

        write_paragraph("اسم المطعم", name)
        write_paragraph("العنوان", address)
        write_paragraph("أرقام الهاتف", phone)
        write_paragraph("ترتيب المطعم", rank)
        write_paragraph("تصنيف المطعم", bestsell)

        write_paragraph("التقييمات", "")
        for info in real_rates.values():
            rate = info.get("rate", "")
            comment = info.get("comment", "")
            write_paragraph("", f"التقييم: {rate}\nالتعليق: {comment}")

        write_paragraph("القائمة الإلكترونية", "")
        for section_name, section in emenu.items():
            write_paragraph("القسم", section_name)
            for product in section.get("products", []):
                write_paragraph(" - الاسم", product.get("name", ""))
                write_paragraph("   الوصف", product.get("desc", ""))
                if 'price' in product:
                    write_paragraph("   السعر", product['price'])
                if 'sizes' in product:
                    for size, price in product['sizes'].items():
                        write_paragraph(f"   الحجم: {size}", price)
                if 'extras' in product:
                    for extra, price in product['extras'].items():
                        write_paragraph(f"   إضافة: {extra}", price)

        write_paragraph("وقت الفتح", str(open_time))
        write_paragraph("وقت الإغلاق", str(close_time))
        write_paragraph("مدة الطهي", str(cooking_time))

        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        docx_path = os.path.join("Restaurants_Word", f"{safe_name}.docx")
        pdf_path = os.path.join("Restaurants_PDF", f"{safe_name}.pdf")
        doc.save(docx_path)

        try:
            convert(docx_path, pdf_path)
            print(f"✅ Converted to PDF: {pdf_path}")
        except Exception as e:
            print(f"❌ Error converting {safe_name} to PDF: {e}")

    if no_eshop_info:
        with open("no_eshop_restaurants.txt", "w", encoding="utf-8") as f:
            for item in no_eshop_info:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("📄 Saved info of restaurants without eShop to: no_eshop_restaurants.txt")

    return {
        "status": "success",
        "pdf_count": len([f for f in os.listdir("Restaurants_PDF") if f.endswith(".pdf")]),
        "no_eshop_count": len(no_eshop_info)
    }

def main():
    parser = argparse.ArgumentParser(description="Generate restaurant PDFs and no-eShop info.")
    parser.add_argument("input_file", help="Path to the input JSON file")

    args = parser.parse_args()

    result = process_restaurants_json(args.input_file)
    print("\n📊 Summary:")
    print(f"✅ PDF files created: {result['pdf_count']}")
    print(f"⚠️ Restaurants without eShop: {result['no_eshop_count']}")

if __name__ == "__main__":
    main()
