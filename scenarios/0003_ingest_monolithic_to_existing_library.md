Given the prompt-mixer library already contains a context.md file.

When the user types:

```
pmx ingest .junie/guidelines.md
```

Then the prompt-mixer library should merge the content of the source document with the existing context.md file, removing duplicate content and incorporating all guidance from both files.