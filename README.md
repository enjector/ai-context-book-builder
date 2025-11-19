# AI Context Book Builder

A workflow and script to capture high-value AI conversations into structured "Books" (PDF/Markdown) for future context injection.

## The Problem

Long context windows are great, but "Context Rot" eventually sets in. Furthermore, valuable insights get trapped in chat logs.

## The Solution

1.  **Prompt** the AI to structure the conversation as a book.
2.  **Save** chapters as individual Markdown files.
3.  **Merge** them using the included script.
4.  **Upload** the resulting PDF to new chats to restore context instantly.

## 1\. The Prompts

**Phase 1: Structure**

> "There's a lot of rich information contained in this conversation, and further research that can be done. Could we create a book covering the principles and how to apply them to real-world problems? The book should cover the end-to-end research. Could you provide a detailed chapter overview of the suggested book, including a title and subtitle, please?"

**Phase 2: Drafting (Iterative)**

> "I will ask you to write each chapter one at a time in detail, using any verification technique like math, examples, or scenarios. This book is for an audience that is either a Software Architect, Business Analyst, or Data Scientist. So, background information on concepts and the domain may be required. Can you begin by writing chapter 1?"

*Continue with each chapter number:*

> "Can you write chapter 2?"

Continue until chapters are completed

**Phase 3: Review**

> "Any other chapters and appendices do you recommend?"

Append the output to 0.md, which is your chapter overview.

*Continue with each chapter number:*

> "Can you write chapter 16?"

AND

> "Can you write Appendix A?"

Continue until chapters and appendecies are completed

## 2\. Usage

### Prerequisites

  * **Node.js & npm** (required for the PDF generator)
  * **markdown-pdf** (The script attempts to install this, but having it ready helps)
    ```bash
    npm install -g markdown-pdf
    ```

  * For Latex support see LATEX.md for pre-requistes

### Steps

1.  Create a folder for your book.
2.  Save AI responses as `0.md` (intro), `1.md`, `2.md` ... `A.md` (appendix).
3.  **Handling Images:** If the AI generates diagrams, save the image to the folder (e.g., `1.1.png`) and update the reference in the markdown file: `![Title](1.1.png)`.
4.  Run the build script (change the name of the output markdown and pdf - default is MyBook.md and MyBook.pdf)

<!-- end list -->

```bash
chmod +x _merge.sh
./_merge.sh
```

This will generate `MyBook.md` and `MyBook.pdf` (or your configured name).

**Upload the resulting Markdown file (or PDF) to new chats to restore context instantly**

Also, you should create a [NotebookLM](https://notebooklm.google.com/) discussion and debate
