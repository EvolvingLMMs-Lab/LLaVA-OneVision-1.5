(() => {
  'use strict';

  const contribEl = document.getElementById('community-contributors');
  const starEl = document.getElementById('community-stargazers');
  if (!contribEl || !starEl) return;

  const elements = {
    contribCount: document.getElementById('community-contrib-count'),
    starCount: document.getElementById('community-star-count'),
    total: document.getElementById('contributor-total'),
    commits: document.getElementById('contributor-commits'),
    active: document.getElementById('contributor-active'),
    search: document.getElementById('contributor-search'),
    sort: document.getElementById('contributor-sort'),
    resultsMeta: document.getElementById('contributor-results-meta'),
    empty: document.getElementById('contributor-empty'),
    starMore: document.getElementById('stargazer-more'),
  };

  const state = {
    contributors: [],
    stargazers: [],
    owner: '',
    repo: '',
    query: '',
    range: 'all',
    sort: 'commits',
    starsVisible: 96,
    hasActivity: false,
  };

  const numberFormat = new Intl.NumberFormat('en-US');
  const DAY_MS = 24 * 60 * 60 * 1000;

  function isChinese() {
    return document.body.classList.contains('lang-zh');
  }

  function localized(en, zh) {
    return isChinese() ? zh : en;
  }

  function create(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function weekTime(week) {
    const value = Date.parse(`${week.week}T00:00:00Z`);
    return Number.isFinite(value) ? value : 0;
  }

  function cutoffFor(range) {
    if (range === 'all') return 0;
    return Date.now() - Number(range) * DAY_MS;
  }

  function weeksInRange(contributor, range) {
    const weeks = Array.isArray(contributor.weekly_activity) ? contributor.weekly_activity : [];
    const cutoff = cutoffFor(range);
    return cutoff ? weeks.filter(week => weekTime(week) >= cutoff) : weeks;
  }

  function commitsInRange(contributor, range) {
    if (range === 'all') return Number(contributor.contributions) || 0;
    return weeksInRange(contributor, range).reduce((sum, week) => sum + (Number(week.commits) || 0), 0);
  }

  function lastActiveTime(contributor) {
    const activeWeeks = (contributor.weekly_activity || []).filter(week => Number(week.commits) > 0);
    return activeWeeks.reduce((latest, week) => Math.max(latest, weekTime(week)), 0);
  }

  function activeWithin(contributor, days) {
    return (contributor.weekly_activity || []).some(
      week => Number(week.commits) > 0 && weekTime(week) >= Date.now() - days * DAY_MS,
    );
  }

  function buildSparkline(contributor) {
    let weeks = weeksInRange(contributor, state.range);
    if (state.range === 'all') weeks = weeks.slice(-52);
    const values = weeks.map(week => Number(week.commits) || 0);
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'contributor-sparkline');
    svg.setAttribute('viewBox', '0 0 120 34');
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.setAttribute('aria-hidden', 'true');

    if (values.length < 2 || Math.max(...values) === 0) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      line.setAttribute('class', 'contributor-sparkline-empty');
      line.setAttribute('d', 'M0 27 L120 27');
      svg.appendChild(line);
      return svg;
    }

    const max = Math.max(...values);
    const points = values.map((value, index) => {
      const x = values.length === 1 ? 60 : (index / (values.length - 1)) * 120;
      const y = 29 - (value / max) * 24;
      return [x, y];
    });
    const linePoints = points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
    const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    area.setAttribute('class', 'contributor-sparkline-area');
    area.setAttribute('d', `M0 32 L${linePoints.replaceAll(' ', ' L')} L120 32 Z`);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    line.setAttribute('class', 'contributor-sparkline-line');
    line.setAttribute('points', linePoints);
    svg.append(area, line);
    return svg;
  }

  function buildContributorCard(contributor) {
    const card = create('a', 'contributor-card');
    card.href = `https://github.com/${encodeURIComponent(contributor.login)}`;
    card.target = '_blank';
    card.rel = 'noopener noreferrer';
    card.setAttribute('role', 'listitem');
    card.setAttribute('aria-label', localized(
      `${contributor.login}, ${contributor.contributions} commits. Open GitHub profile.`,
      `${contributor.login}，${contributor.contributions} 次提交。打开 GitHub 主页。`,
    ));

    const header = create('div', 'contributor-card-header');
    const avatarWrap = create('div', 'contributor-avatar-wrap');
    const avatar = create('img', 'contributor-card-avatar');
    avatar.src = contributor.avatar_url;
    avatar.alt = '';
    avatar.loading = 'lazy';
    avatar.decoding = 'async';
    avatarWrap.appendChild(avatar);
    if (contributor.rank <= 3) {
      avatarWrap.appendChild(create('span', `contributor-rank contributor-rank-${contributor.rank}`, `#${contributor.rank}`));
    }

    const identity = create('div', 'contributor-identity');
    identity.appendChild(create('strong', 'contributor-login', contributor.login));
    const status = create('span', activeWithin(contributor, 90) ? 'contributor-status active' : 'contributor-status');
    status.appendChild(create('span', 'contributor-status-dot'));
    status.appendChild(document.createTextNode(activeWithin(contributor, 90)
      ? localized('Active recently', '近期活跃')
      : localized('Contributor', '项目贡献者')));
    identity.appendChild(status);
    header.append(avatarWrap, identity);

    const selectedCommits = commitsInRange(contributor, state.range);
    const stats = create('div', 'contributor-card-stats');
    const primaryStat = create('div', 'contributor-card-stat primary');
    primaryStat.append(
      create('strong', '', numberFormat.format(selectedCommits)),
      create('span', '', state.range === 'all' ? localized('commits', '次提交') : localized('in period', '周期内提交')),
    );
    const recentStat = create('div', 'contributor-card-stat');
    const lastActive = lastActiveTime(contributor);
    recentStat.append(
      create('strong', '', lastActive ? new Intl.DateTimeFormat(isChinese() ? 'zh-CN' : 'en', { month: 'short', year: 'numeric' }).format(lastActive) : '—'),
      create('span', '', localized('last active', '最近活跃')),
    );
    stats.append(primaryStat, recentStat);

    const trend = create('div', 'contributor-trend');
    trend.append(
      create('span', 'contributor-trend-label', localized('Weekly activity', '每周活跃度')),
      buildSparkline(contributor),
    );
    const arrow = create('span', 'contributor-card-arrow', '↗');
    arrow.setAttribute('aria-hidden', 'true');
    card.append(header, stats, trend, arrow);
    return card;
  }

  function visibleContributors() {
    const query = state.query.trim().toLowerCase();
    let contributors = state.contributors.filter(contributor => contributor.login.toLowerCase().includes(query));
    if (state.range !== 'all' && state.hasActivity) {
      contributors = contributors.filter(contributor => commitsInRange(contributor, state.range) > 0);
    }
    return contributors.sort((a, b) => {
      if (state.sort === 'name') return a.login.localeCompare(b.login, undefined, { sensitivity: 'base' });
      if (state.sort === 'recent') return lastActiveTime(b) - lastActiveTime(a) || b.contributions - a.contributions;
      return commitsInRange(b, state.range) - commitsInRange(a, state.range) || a.login.localeCompare(b.login);
    });
  }

  function updateSortLabels() {
    if (!elements.sort) return;
    const labels = isChinese()
      ? ['提交数最多', '最近活跃', '名称 A–Z']
      : ['Most commits', 'Recently active', 'Name A–Z'];
    Array.from(elements.sort.options).forEach((option, index) => { option.textContent = labels[index]; });
  }

  function renderContributors() {
    const contributors = visibleContributors();
    const totalCommits = state.range === 'all'
      ? state.contributors.reduce((sum, contributor) => sum + (Number(contributor.contributions) || 0), 0)
      : state.contributors.reduce((sum, contributor) => sum + commitsInRange(contributor, state.range), 0);
    const active90 = state.contributors.filter(contributor => activeWithin(contributor, 90)).length;

    elements.total.textContent = numberFormat.format(state.contributors.length);
    elements.commits.textContent = numberFormat.format(totalCommits);
    elements.active.textContent = state.hasActivity ? numberFormat.format(active90) : '—';
    elements.empty.hidden = contributors.length !== 0;
    elements.resultsMeta.textContent = localized(
      `${contributors.length} of ${state.contributors.length} contributors shown`,
      `显示 ${contributors.length} / ${state.contributors.length} 位贡献者`,
    );

    const fragment = document.createDocumentFragment();
    contributors.forEach(contributor => fragment.appendChild(buildContributorCard(contributor)));
    contribEl.replaceChildren(fragment);
  }

  function buildStarTile(stargazer, index) {
    const link = create('a', `community-tile${index === 0 ? ' community-tile-top' : ''}`);
    link.href = `https://github.com/${encodeURIComponent(stargazer.login)}`;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.title = stargazer.starred_at
      ? `${stargazer.login} · ${new Date(stargazer.starred_at).toLocaleDateString()}`
      : stargazer.login;
    link.setAttribute('aria-label', stargazer.login);
    const img = create('img', 'community-avatar');
    img.alt = '';
    img.loading = 'lazy';
    img.decoding = 'async';
    img.src = stargazer.avatar_url;
    link.appendChild(img);
    return link;
  }

  function renderStars() {
    const fragment = document.createDocumentFragment();
    state.stargazers.slice(0, state.starsVisible).forEach((star, index) => fragment.appendChild(buildStarTile(star, index)));
    starEl.replaceChildren(fragment);
    elements.starMore.hidden = state.starsVisible >= state.stargazers.length;
  }

  function showError(error) {
    console.error('community.json load failed:', error);
    const message = create('p', 'community-error', localized('Community data unavailable.', '社区数据暂时不可用。'));
    contribEl.replaceChildren(message);
    starEl.replaceChildren(message.cloneNode(true));
  }

  document.querySelectorAll('.contributor-range-button').forEach(button => {
    button.setAttribute('aria-pressed', String(button.classList.contains('active')));
    button.addEventListener('click', () => {
      state.range = button.dataset.range || 'all';
      document.querySelectorAll('.contributor-range-button').forEach(item => {
        const active = item === button;
        item.classList.toggle('active', active);
        item.setAttribute('aria-pressed', String(active));
      });
      renderContributors();
    });
  });

  elements.search?.addEventListener('input', event => {
    state.query = event.target.value;
    renderContributors();
  });
  elements.sort?.addEventListener('change', event => {
    state.sort = event.target.value;
    renderContributors();
  });
  elements.starMore?.addEventListener('click', () => {
    state.starsVisible += 96;
    renderStars();
  });
  window.addEventListener('llava-language-change', () => {
    updateSortLabels();
    if (state.contributors.length) renderContributors();
  });

  fetch('../assets/community.json', { cache: 'no-cache' })
    .then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then(data => {
      state.owner = data.owner;
      state.repo = data.repo;
      state.contributors = (data.contributors || []).map((contributor, index) => ({ ...contributor, rank: index + 1 }));
      state.stargazers = data.stargazers || [];
      state.hasActivity = state.contributors.some(contributor => (contributor.weekly_activity || []).length > 0);

      if (!state.hasActivity) {
        document.querySelectorAll('.contributor-range-button:not([data-range="all"])').forEach(button => {
          button.disabled = true;
          button.title = localized('Activity trends are being prepared by GitHub.', 'GitHub 正在准备活动趋势数据。');
        });
      }
      if (elements.contribCount) {
        elements.contribCount.textContent = numberFormat.format(state.contributors.length);
        elements.contribCount.href = `https://github.com/${state.owner}/${state.repo}/graphs/contributors`;
      }
      if (elements.starCount) {
        elements.starCount.textContent = numberFormat.format(state.stargazers.length);
        elements.starCount.href = `https://github.com/${state.owner}/${state.repo}/stargazers`;
      }
      updateSortLabels();
      renderContributors();
      renderStars();
    })
    .catch(showError);
})();
