/* Shared, isolated navigation for the local administration pages. */
(() => {
  const sections = [
    { label: '总览', href: '/admin', key: 'admin' },
    { label: '知识库', href: '/kb', key: 'kb' },
    { label: '问题池', href: '/topics', key: 'topics' },
    { label: '审核队列', href: '/review', key: 'review' },
    { label: '验收看板', href: '/acceptance', key: 'acceptance' },
    { label: 'RAG 评估', href: '/rag-eval', key: 'rag-eval' },
  ];
  const subpages = {
    kb: [
      ['库存', '#inventory'], ['手工录入', '#ingest'], ['建库材料', '#sources'],
      ['对话挖知识', '#mining'], ['向量化', '#vectors'], ['检索自测', '#search'], ['最近入库', '#recent'],
    ],
  };
  const styleId = 'petshop-admin-nav-style';

  function mountAdminNav(active) {
    if (!document.getElementById(styleId)) {
      const style = document.createElement('style');
      style.id = styleId;
      style.textContent = `
        .ps-admin-nav{background:#173c34;color:#f7f8f2;font:14px/1.45 system-ui,"Microsoft YaHei",sans-serif}
        .ps-admin-nav *{box-sizing:border-box}.ps-admin-nav__inner{max-width:1200px;margin:auto;padding:13px 22px 10px}
        .ps-admin-nav__top{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap}
        .ps-admin-nav__brand{font-weight:800;letter-spacing:.02em;color:white;text-decoration:none;font-size:18px}
        .ps-admin-nav__brand small{font-size:11px;font-weight:500;color:#b9d8cb;display:block;letter-spacing:.13em}
        .ps-admin-nav__links,.ps-admin-nav__sub{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
        .ps-admin-nav__links a,.ps-admin-nav__sub a{color:#d9e9df;text-decoration:none;border-radius:8px;padding:7px 10px}
        .ps-admin-nav__links a:hover,.ps-admin-nav__sub a:hover{background:#2a5d4e;color:white}
        .ps-admin-nav__links a[aria-current="page"]{background:#dff1d9;color:#173c34;font-weight:700}
        .ps-admin-nav__sub{border-top:1px solid #42695d;margin-top:11px;padding-top:8px}
        .ps-admin-nav__sub a{font-size:12px;padding:4px 9px;color:#b9d8cb}
        .ps-admin-nav__sub a[aria-current="location"]{color:white;background:#2a5d4e}
        @media(max-width:640px){.ps-admin-nav__inner{padding:12px 14px}.ps-admin-nav__links{overflow-x:auto;flex-wrap:nowrap;width:100%}.ps-admin-nav__links a{white-space:nowrap}}
      `;
      document.head.appendChild(style);
    }
    const old = document.getElementById('petshop-admin-nav');
    if (old) old.remove();
    const nav = document.createElement('nav');
    nav.id = 'petshop-admin-nav'; nav.className = 'ps-admin-nav'; nav.setAttribute('aria-label', '后台导航');
    const inner = document.createElement('div'); inner.className = 'ps-admin-nav__inner';
    const top = document.createElement('div'); top.className = 'ps-admin-nav__top';
    const brand = document.createElement('a'); brand.className = 'ps-admin-nav__brand'; brand.href = '/admin';
    brand.textContent = 'PetShop Helper';
    const small = document.createElement('small'); small.textContent = '本地管理工作台'; brand.appendChild(small); top.appendChild(brand);
    const links = document.createElement('div'); links.className = 'ps-admin-nav__links';
    for (const item of sections) {
      const link = document.createElement('a'); link.href = item.href; link.textContent = item.label;
      if (item.key === active) link.setAttribute('aria-current', 'page');
      links.appendChild(link);
    }
    top.appendChild(links); inner.appendChild(top);
    if (subpages[active]) {
      const sub = document.createElement('div'); sub.className = 'ps-admin-nav__sub'; sub.setAttribute('aria-label', '知识库分区');
      for (const [label, href] of subpages[active]) {
        const link = document.createElement('a'); link.href = href; link.textContent = label;
        sub.appendChild(link);
      }
      inner.appendChild(sub);
      const update = () => {
        for (const link of sub.querySelectorAll('a')) {
          if (link.getAttribute('href') === location.hash) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        }
      };
      window.addEventListener('hashchange', update); update();
    }
    nav.appendChild(inner);
    const host = document.getElementById('adminNav');
    if (host) host.appendChild(nav); else document.body.prepend(nav);
  }
  window.mountAdminNav = mountAdminNav;
})();
