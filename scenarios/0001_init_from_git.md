Given the user has installed prompt-mixer, and there is no existing prompt-mixer configuration.

When the user types:

```
pmx init --remote git@github.com:svetzal/prompt-mixer-sample.git
```

Then prompt-mixer should clone the repo into its default location.