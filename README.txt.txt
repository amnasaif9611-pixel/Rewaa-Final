Project Name: رِواء

Description:
رِواء منصة خليجية ذكية للأمن المائي. تعرض بيانات تجريبية لاستهلاك المياه، تحلل السيناريوهات، وتستخدم OpenAI لتحويل المؤشرات الرقمية إلى تفسيرات وتوصيات لصانع القرار.

How to Run:
1. Install requirements:
   pip install -r requirements.txt

2. Add your OpenAI API key:
   - Local Streamlit: create .streamlit/secrets.toml
   - Add: OPENAI_API_KEY = "your-key"
   - Never upload the real secrets.toml to GitHub.

3. Run:
   streamlit run app.py

Notes:
- The included CSV is demonstration data, not live government data.
- When no API key is configured, Rewaa clearly switches to a local demo fallback.
