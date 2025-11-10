"""
Create a simple test PDF with equations and tables
This allows us to test Marker quickly without needing access to Zotero
"""

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    def create_test_pdf(filename="test_sample.pdf"):
        """Create a simple test PDF with scientific content"""
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(inch, height - inch, "Test Scientific Paper")

        # Subtitle
        c.setFont("Helvetica", 12)
        c.drawString(inch, height - 1.3*inch, "A sample document for testing Marker PDF extraction")

        # Abstract section
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, height - 2*inch, "Abstract")

        c.setFont("Helvetica", 10)
        text = "This is a test document containing mathematical equations and tabular data. "
        text += "The purpose is to evaluate the extraction capabilities of different PDF parsers."

        # Wrap text
        y_position = height - 2.3*inch
        for line in [text[i:i+80] for i in range(0, len(text), 80)]:
            c.drawString(inch, y_position, line)
            y_position -= 0.2*inch

        # Equation section (simulated)
        y_position -= 0.3*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, y_position, "Mathematical Equation")

        y_position -= 0.3*inch
        c.setFont("Helvetica-Oblique", 12)
        c.drawString(inch + 0.5*inch, y_position, "E = mc²")

        # Table section
        y_position -= 0.5*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, y_position, "Table 1: Sample Data")

        # Draw simple table
        y_position -= 0.3*inch
        c.setFont("Helvetica", 10)

        # Table headers
        c.drawString(inch, y_position, "Parameter")
        c.drawString(inch + 2*inch, y_position, "Value")
        c.drawString(inch + 3.5*inch, y_position, "Unit")

        # Table rows
        data = [
            ("Temperature", "25.3", "°C"),
            ("Pressure", "101.3", "kPa"),
            ("Humidity", "65", "%")
        ]

        y_position -= 0.25*inch
        for param, value, unit in data:
            c.drawString(inch, y_position, param)
            c.drawString(inch + 2*inch, y_position, value)
            c.drawString(inch + 3.5*inch, y_position, unit)
            y_position -= 0.2*inch

        # Conclusion
        y_position -= 0.3*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, y_position, "Conclusion")

        y_position -= 0.3*inch
        c.setFont("Helvetica", 10)
        c.drawString(inch, y_position, "This test PDF demonstrates basic scientific document structure.")

        c.save()
        print(f"✓ Created test PDF: {filename}")
        return filename

    if __name__ == "__main__":
        create_test_pdf()

except ImportError:
    print("❌ reportlab not installed. Creating fallback message...")
    print("Install with: pip install reportlab")
    print("\nAlternatively, you can test Marker with any existing PDF from your system.")
