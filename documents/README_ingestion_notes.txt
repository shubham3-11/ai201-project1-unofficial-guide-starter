SOURCE: Project notes
TYPE: ingestion_readme
URL: local
TITLE: Notes for Milestone 3 document ingestion
---
[WHAT IS INCLUDED]
This documents folder contains 10 source files named with numeric prefixes to match the planning table. Each file uses a simple header block followed by a --- separator and plain UTF-8 text content.

[URL CLEANUP]
The source URLs have been cleaned to remove tracking parameters such as ?utm_source=chatgpt.com.

[IMPORTANT CORRECTION]
The Rutgers FAQ source uses the corrected Rutgers Off-Campus Living FAQ URL:
https://ruoffcampus.rutgers.edu/campus-living/faqs

[QUALITY NOTE]
Some pages, especially Reddit and listing pages, expose different amounts of content depending on login state, collapsed comments, or JavaScript rendering. These files are good starter ingestion documents, but for a stronger RAG dataset you should manually add more useful comments from the live pages where needed.

[RECOMMENDED LOADER BEHAVIOR]
- Read all documents/*.txt files.
- Split metadata from body at the first --- line.
- Parse SOURCE, TYPE, URL, and TITLE from the header.
- Normalize whitespace.
- Chunk the body using about 500 characters with 100 characters overlap.
- Attach parsed metadata to every chunk.
