(function () {
  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function formatDate(value) {
    if (!value) return '';
    var d = new Date(value);
    if (isNaN(d.getTime())) return value;
    return d.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function textFromHtml(html) {
    if (!html) return '';
    var div = document.createElement('div');
    div.innerHTML = html;
    return (div.textContent || div.innerText || '').trim();
  }

  function buildItem(item) {
    var title = item.querySelector('title');
    var link = item.querySelector('link');
    var pubDate = item.querySelector('pubDate') || item.querySelector('updated');
    var description = item.querySelector('description') || item.querySelector('summary');

    var li = document.createElement('li');
    li.className = 'ai-feed-item';

    var a = document.createElement('a');
    a.className = 'ai-feed-link';
    a.href = link ? link.textContent.trim() : '#';
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = title ? title.textContent.trim() : 'Untitled';

    var meta = document.createElement('div');
    meta.className = 'ai-feed-item-meta';
    meta.textContent = formatDate(pubDate ? pubDate.textContent.trim() : '');

    li.appendChild(a);
    li.appendChild(meta);

    if (description && description.textContent) {
      var summaryText = textFromHtml(description.textContent);
      if (summaryText) {
        var summary = document.createElement('div');
        summary.className = 'ai-feed-item-summary';
        summary.textContent = summaryText;
        li.appendChild(summary);
      }
    }

    return li;
  }

  function renderFeed(xmlText, container) {
    var parser = new DOMParser();
    var xml = parser.parseFromString(xmlText, 'text/xml');

    var error = xml.querySelector('parsererror');
    if (error) {
      throw new Error('Feed XML parse error');
    }

    var channel = xml.querySelector('channel');
    var lastBuild = channel ? channel.querySelector('lastBuildDate') : null;
    var updated = channel ? channel.querySelector('pubDate') : null;

    var meta = $('#ai-feed-meta', container);
    if (meta) {
      var dateText = (lastBuild && lastBuild.textContent) || (updated && updated.textContent) || '';
      meta.textContent = dateText ? 'Last updated ' + formatDate(dateText) : 'Latest updates';
    }

    var list = $('#ai-feed-list', container);
    if (!list) return;

    list.innerHTML = '';

    var items = xml.querySelectorAll('item');
    if (!items.length) {
      var empty = document.createElement('li');
      empty.className = 'ai-feed-empty';
      empty.textContent = 'No items in the feed yet.';
      list.appendChild(empty);
      return;
    }

    items.forEach(function (item) {
      list.appendChild(buildItem(item));
    });
  }

  function showError(container, message) {
    var error = $('#ai-feed-error', container);
    if (!error) return;
    error.style.display = 'block';
    error.textContent = message;
  }

  function init() {
    var container = document.querySelector('.ai-feed');
    if (!container) return;

    var feedUrl = container.getAttribute('data-feed-url');
    if (!feedUrl) {
      showError(container, 'Feed URL is missing.');
      return;
    }

    fetch(feedUrl, { cache: 'no-store' })
      .then(function (res) {
        if (!res.ok) throw new Error('Failed to load feed');
        return res.text();
      })
      .then(function (text) {
        renderFeed(text, container);
      })
      .catch(function () {
        showError(container, 'Unable to load the AI feed right now.');
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
