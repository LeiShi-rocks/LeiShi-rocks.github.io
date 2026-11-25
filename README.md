# LeiShi-rocks.github.io
A repository for my git pages

## Blog

The blog index is authored in [blog.jemdoc](blog.jemdoc) and published as
[blog.html](blog.html). Articles live in `blog/<slug>/index.html`, with their
figures and data alongside them. The shared [blog/style.css](blog/style.css)
styles the index and article navigation; individual posts can retain their own
reading layout and interactive figures.

To add a post, create its folder, include `../style.css`, and reuse the navigation
from [blog/gauger/index.html](blog/gauger/index.html). Add an article entry to
[blog.jemdoc](blog.jemdoc), then rebuild from the repository root:

```sh
python3 jemdoc -c mysite.conf blog.jemdoc
```

When changing [MENU](MENU), regenerate the root jemdoc pages so the published
navigation stays in sync:

```sh
python3 jemdoc -c mysite.conf *.jemdoc
```

Commit both the jemdoc source and generated HTML. No server is needed to preview
the blog: open [blog.html](blog.html) in a browser.
