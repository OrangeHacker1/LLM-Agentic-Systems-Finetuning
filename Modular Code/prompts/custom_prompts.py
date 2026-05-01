# prompts/custom_prompts.py

CUSTOM_PROMPTS = {

    "extraction": [
        # EASY (10)
        "Extract name, location, and date from: John visited Paris on July 5th.",
        "Extract entities from: Alice went to Tokyo on March 3rd.",
        "Extract details from: Bob traveled to NYC on Jan 1.",
        "Find name/location/date: Sarah visited London in June.",
        "Extract structured info: Mike went to Berlin on Feb 2.",
        "Extract details: Emma visited Rome in April.",
        "Extract entities: Tom traveled to Madrid on May 6.",
        "Extract info: Lisa visited Dubai in August.",
        "Extract details: Jack went to Toronto in September.",
        "Extract info: Anna visited Sydney in October.",

        # MEDIUM (20)
        "From text extract name/location/date: John and Mary visited Paris on July 5th.",
        "Extract structured entities from paragraph: Alice traveled across Tokyo and Kyoto in March.",
        "Extract key fields: Bob moved from LA to NYC in 2020.",
        "Extract entities from: Sarah visited multiple cities including London and Paris.",
        "Extract name/location/date from mixed text with noise: ### Mike Berlin Feb 2 ###",
        "Extract structured info from paragraph with multiple dates.",
        "Extract entities even if order is scrambled.",
        "Extract fields even if date format varies.",
        "Extract info from paragraph with extra irrelevant text.",
        "Extract structured data from informal sentence.",

        # HARD (20)
        "Extract structured JSON from messy paragraph with multiple names and dates.",
        "Extract entities from long paragraph with distractions.",
        "Extract primary subject name/location/date ignoring irrelevant data.",
        "Extract structured fields from noisy OCR-like text.",
        "Extract entities from ambiguous sentence.",
        "Extract info when multiple candidates exist.",
        "Extract main event details from paragraph.",
        "Extract structured data from mixed language text.",
        "Extract key info from narrative paragraph.",
        "Extract entities from multi-sentence paragraph.",
    ],

    "classification": [
        # EASY
        "Classify: The evidence supports the claim.",
        "Classify: This is false.",
        "Classify: Not enough information.",
        "Label as supported/refuted/neutral: Evidence confirms hypothesis.",
        "Classify: The claim is wrong.",

        # MEDIUM
        "Classify with reasoning: The data partially supports the claim.",
        "Determine label: Mixed evidence exists.",
        "Classify ambiguous statement.",
        "Classify nuanced claim.",
        "Classify uncertain statement.",

        # HARD
        "Classify paragraph with conflicting evidence.",
        "Classify long argument.",
        "Determine label from complex reasoning.",
        "Classify scientific claim.",
        "Classify multi-sentence argument.",
    ],

    "schema": [
        "Generate JSON for: Product iPhone costs $999 by Apple.",
        "Convert to schema: Tesla Model S costs $80,000.",
        "Create JSON object for: Book titled X by author Y.",
        "Generate structured data for company description.",
        "Convert paragraph into structured JSON.",
        "Generate JSON for nested schema.",
        "Create structured object with multiple attributes.",
        "Generate JSON from semi-structured text.",
        "Convert product description into schema.",
        "Generate structured JSON from messy description.",
    ],

    "repair": [
        "Fix JSON: {name: John, location: Paris}",
        "Repair: {'city': 'Austin', 'temp': 75,}",
        "Fix malformed JSON with missing quotes.",
        "Correct JSON with trailing commas.",
        "Repair nested broken JSON.",
        "Fix JSON with wrong types.",
        "Repair JSON missing brackets.",
        "Fix JSON with duplicated keys.",
        "Correct JSON formatting errors.",
        "Repair invalid JSON string.",
    ],

    "tool": [
        "Generate arguments: Weather in Austin tomorrow.",
        "Call function: get_weather for NYC Monday.",
        "Create JSON for weather API.",
        "Generate tool arguments from query.",
        "Produce JSON for function call.",
        "Generate API call arguments.",
        "Convert question to tool JSON.",
        "Create structured function parameters.",
        "Generate JSON for API input.",
        "Produce tool call JSON.",
    ]
}