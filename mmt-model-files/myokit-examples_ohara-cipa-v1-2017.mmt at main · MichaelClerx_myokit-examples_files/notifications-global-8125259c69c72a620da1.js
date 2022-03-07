"use strict";(()=>{var it=Object.defineProperty;var a=(k,P)=>it(k,"name",{value:P,configurable:!0});(globalThis.webpackChunk=globalThis.webpackChunk||[]).push([["notifications-global","github_form_ts-github_navigation_ts"],{42650:(k,P,b)=>{b.d(P,{v:()=>_,H:()=>v});var o=b(11427);function _(){const l=document.getElementById("ajax-error-message");l&&(l.hidden=!1)}a(_,"showGlobalError");function v(){const l=document.getElementById("ajax-error-message");l&&(l.hidden=!0)}a(v,"hideGlobalError"),(0,o.on)("deprecatedAjaxError","[data-remote]",function(l){const c=l.detail,{error:A,text:g}=c;l.currentTarget===l.target&&(A==="abort"||A==="canceled"||(/<html/.test(g)?(_(),l.stopImmediatePropagation()):setTimeout(function(){l.defaultPrevented||_()},0)))}),(0,o.on)("deprecatedAjaxSend","[data-remote]",function(){v()}),(0,o.on)("click",".js-ajax-error-dismiss",function(){v()})},83895:(k,P,b)=>{b.d(P,{Ty:()=>_,YE:()=>v,Zf:()=>l});var o=b(13178);const _=a(()=>{const c=document.querySelector("meta[name=keyboard-shortcuts-preference]");return c?c.content==="all":!0},"areCharacterKeyShortcutsEnabled"),v=a(c=>/Enter|Arrow|Escape|Meta|Control|Esc/.test(c)||c.includes("Alt")&&c.includes("Shift"),"isNonCharacterKeyShortcut"),l=a(c=>{const A=(0,o.EL)(c);return _()?!0:v(A)},"isShortcutAllowed")},68559:(k,P,b)=>{b.d(P,{cv:()=>o,VZ:()=>_,_C:()=>v,oE:()=>l});function o(g){const f=g.getBoundingClientRect();return{top:f.top+window.pageYOffset,left:f.left+window.pageXOffset}}a(o,"offset");function _(g){let f=g;const d=f.ownerDocument;if(!d||!f.offsetParent)return;const y=d.defaultView.HTMLElement;if(f!==d.body){for(;f!==d.body;){if(f.parentElement instanceof y)f=f.parentElement;else return;const{position:T,overflowY:D,overflowX:p}=getComputedStyle(f);if(T==="fixed"||D==="auto"||p==="auto"||D==="scroll"||p==="scroll")break}return f instanceof Document?null:f}}a(_,"overflowParent");function v(g,f){let d=f;const y=g.ownerDocument;if(!y)return;const T=y.documentElement;if(!T||g===T)return;const D=l(g,d);if(!D)return;d=D._container;const p=d===y.documentElement&&y.defaultView?{top:y.defaultView.pageYOffset,left:y.defaultView.pageXOffset}:{top:d.scrollTop,left:d.scrollLeft},L=D.top-p.top,x=D.left-p.left,j=d.clientHeight,H=d.clientWidth,K=j-(L+g.offsetHeight),R=H-(x+g.offsetWidth);return{top:L,left:x,bottom:K,right:R,height:j,width:H}}a(v,"overflowOffset");function l(g,f){let d=g;const y=d.ownerDocument;if(!y)return;const T=y.documentElement;if(!T)return;const D=y.defaultView.HTMLElement;let p=0,L=0;const x=d.offsetHeight,j=d.offsetWidth;for(;!(d===y.body||d===f);)if(p+=d.offsetTop||0,L+=d.offsetLeft||0,d.offsetParent instanceof D)d=d.offsetParent;else return;let H,K,R;if(!f||f===y||f===y.defaultView||f===y.documentElement||f===y.body)R=T,H=c(y.body,T),K=A(y.body,T);else if(f instanceof D)R=f,H=f.scrollHeight,K=f.scrollWidth;else return;const Z=H-(p+x),F=K-(L+j);return{top:p,left:L,bottom:Z,right:F,_container:R}}a(l,"positionedOffset");function c(g,f){return Math.max(g.scrollHeight,f.scrollHeight,g.offsetHeight,f.offsetHeight,f.clientHeight)}a(c,"getDocumentHeight");function A(g,f){return Math.max(g.scrollWidth,f.scrollWidth,g.offsetWidth,f.offsetWidth,f.clientWidth)}a(A,"getDocumentWidth")},16393:(k,P,b)=>{b.d(P,{a:()=>v,D:()=>l});var o=b(7358),_=b(75914);async function v(c,A,g){const f=new Request(A,g);f.headers.append("X-Requested-With","XMLHttpRequest");const d=await self.fetch(f);if(d.status<200||d.status>=300)throw new Error(`HTTP ${d.status}${d.statusText||""}`);return(0,o.t)((0,o.P)(c),d),(0,_.r)(c,await d.text())}a(v,"fetchSafeDocumentFragment");function l(c,A,g=1e3){return a(async function f(d){const y=new Request(c,A);y.headers.append("X-Requested-With","XMLHttpRequest");const T=await self.fetch(y);if(T.status<200||T.status>=300)throw new Error(`HTTP ${T.status}${T.statusText||""}`);if(T.status===200)return T;if(T.status===202)return await new Promise(D=>setTimeout(D,d)),f(d*1.5);throw new Error(`Unexpected ${T.status} response status from poll endpoint`)},"poll")(g)}a(l,"fetchPoll")},65889:(k,P,b)=>{b.d(P,{Bt:()=>c,Se:()=>g,DN:()=>f,sw:()=>d,KL:()=>T,qC:()=>D});var o=b(11427),_=b(27316),v=b(42650);(0,o.on)("click",".js-remote-submit-button",async function(p){const x=p.currentTarget.form;p.preventDefault();let j;try{j=await fetch(x.action,{method:x.method,body:new FormData(x),headers:{Accept:"application/json","X-Requested-With":"XMLHttpRequest"}})}catch{}j&&!j.ok&&(0,v.v)()});function l(p,L,x){return p.dispatchEvent(new CustomEvent(L,{bubbles:!0,cancelable:x}))}a(l,"fire");function c(p,L){L&&(A(p,L),(0,_.j)(L)),l(p,"submit",!0)&&p.submit()}a(c,"requestSubmit");function A(p,L){if(!(p instanceof HTMLFormElement))throw new TypeError("The specified element is not of type HTMLFormElement.");if(!(L instanceof HTMLElement))throw new TypeError("The specified element is not of type HTMLElement.");if(L.type!=="submit")throw new TypeError("The specified element is not a submit button.");if(!p||p!==L.form)throw new Error("The specified element is not owned by the form element.")}a(A,"checkButtonValidity");function g(p,L){if(typeof L=="boolean")if(p instanceof HTMLInputElement)p.checked=L;else throw new TypeError("only checkboxes can be set to boolean value");else{if(p.type==="checkbox")throw new TypeError("checkbox can't be set to string value");p.value=L}l(p,"change",!1)}a(g,"changeValue");function f(p,L){for(const x in L){const j=L[x],H=p.elements.namedItem(x);(H instanceof HTMLInputElement||H instanceof HTMLTextAreaElement)&&(H.value=j)}}a(f,"fillFormValues");function d(p){if(!(p instanceof HTMLElement))return!1;const L=p.nodeName.toLowerCase(),x=(p.getAttribute("type")||"").toLowerCase();return L==="select"||L==="textarea"||L==="input"&&x!=="submit"&&x!=="reset"||p.isContentEditable}a(d,"isFormField");function y(p){return new URLSearchParams(p)}a(y,"searchParamsFromFormData");function T(p,L){const x=new URLSearchParams(p.search),j=y(L);for(const[H,K]of j)x.append(H,K);return x.toString()}a(T,"combineGetFormSearchParams");function D(p){return y(new FormData(p)).toString()}a(D,"serialize")},7358:(k,P,b)=>{b.d(P,{P:()=>o,t:()=>v});function o(l){const c=[...l.querySelectorAll("meta[name=html-safe-nonce]")].map(A=>A.content);if(c.length<1)throw new Error("could not find html-safe-nonce on document");return c}a(o,"getDocumentHtmlSafeNonces");class _ extends Error{constructor(c,A){super(`${c} for HTTP ${A.status}`);this.response=A}}a(_,"ResponseError");function v(l,c,A=!1){const g=c.headers.get("content-type")||"";if(!A&&!g.startsWith("text/html"))throw new _(`expected response with text/html, but was ${g}`,c);if(A&&!(g.startsWith("text/html")||g.startsWith("application/json")))throw new _(`expected response with text/html or application/json, but was ${g}`,c);const f=c.headers.get("x-html-safe");if(f){if(!l.includes(f))throw new _("response X-HTML-Safe nonce did not match",c)}else throw new _("missing X-HTML-Safe nonce",c)}a(v,"verifyResponseHtmlSafeNonce")},36738:(k,P,b)=>{b.d(P,{QZ:()=>R,ZH:()=>G,jK:()=>Z,T_:()=>Q,Sw:()=>tt,VF:()=>V,VH:()=>U});var o=b(27157),_=b(11427),v=b(68559),l=b(13178),c=b(83895),A=b(30762);function g(i,n){let m=i;const h=i.ownerDocument;(m===h||m===h.defaultView||m===h.documentElement||m===h.body)&&(m=h);const C=h.defaultView.Document;if(m instanceof C){const M=n.top!=null?n.top:h.defaultView.pageYOffset,W=n.left!=null?n.left:h.defaultView.pageXOffset;h.defaultView.scrollTo(W,M);return}const S=h.defaultView.HTMLElement;if(!(m instanceof S))throw new Error("invariant");m.scrollTop=n.top,n.left!=null&&(m.scrollLeft=n.left)}a(g,"scrollTo");var f=b(44088);const d=navigator.userAgent.match(/Macintosh/),y=d?"metaKey":"ctrlKey",T=d?"Meta":"Control";let D=!1,p={x:0,y:0};(0,A.N7)(".js-navigation-container:not(.js-navigation-container-no-mouse)",{subscribe:i=>(0,o.qC)((0,o.RB)(i,"mouseover",L),(0,o.RB)(i,"mouseover",x))});function L(i){i instanceof MouseEvent&&((p.x!==i.clientX||p.y!==i.clientY)&&(D=!1),p={x:i.clientX,y:i.clientY})}a(L,"onContainerMouseMove");function x(i){if(D)return;const n=i.currentTarget,{target:m}=i;if(!(m instanceof Element)||!(n instanceof HTMLElement)||!n.closest(".js-active-navigation-container"))return;const h=m.closest(".js-navigation-item");h&&N(h,n)}a(x,"onContainerMouseOver");let j=0;(0,A.N7)(".js-active-navigation-container",{add(){j++,j===1&&document.addEventListener("keydown",H)},remove(){j--,j===0&&document.removeEventListener("keydown",H)}});function H(i){if(i.target!==document.body&&i.target instanceof HTMLElement&&!i.target.classList.contains("js-navigation-enable"))return;D=!0;const n=z();let m=!1;if(n){const h=n.querySelector(".js-navigation-item.navigation-focus")||n;m=(0,_.f)(h,"navigation:keydown",{hotkey:(0,l.EL)(i),originalEvent:i,originalTarget:i.target})}m||i.preventDefault()}a(H,"fireCustomKeydown"),(0,_.on)("navigation:keydown",".js-active-navigation-container",function(i){const n=i.currentTarget,m=i.detail.originalTarget.matches("input, textarea"),h=i.target;if(!!(0,c.Zf)(i.detail.originalEvent)){if(h.classList.contains("js-navigation-item"))if(m){if(d)switch((0,l.EL)(i.detail.originalEvent)){case"Control+n":q(h,n);break;case"Control+p":B(h,n)}switch((0,l.EL)(i.detail.originalEvent)){case"ArrowUp":B(h,n);break;case"ArrowDown":q(h,n);break;case"Enter":case`${T}+Enter`:X(h,i.detail.originalEvent[y]);break}}else{if(d)switch((0,l.EL)(i.detail.originalEvent)){case"Control+n":q(h,n);break;case"Control+p":B(h,n);break;case"Alt+v":et(h,n);break;case"Control+v":Y(h,n)}switch((0,l.EL)(i.detail.originalEvent)){case"j":case"J":q(h,n);break;case"k":case"K":B(h,n);break;case"o":case"Enter":case`${T}+Enter`:X(h,i.detail[y]);break}}else{const C=$(n)[0];if(C)if(m){if(d)switch((0,l.EL)(i.detail.originalEvent)){case"Control+n":N(C,n)}switch((0,l.EL)(i.detail.originalEvent)){case"ArrowDown":N(C,n)}}else{if(d)switch((0,l.EL)(i.detail.originalEvent)){case"Control+n":case"Control+v":N(C,n)}switch((0,l.EL)(i.detail.originalEvent)){case"j":N(C,n)}}}if(m){if(d)switch((0,l.EL)(i.detail.originalEvent)){case"Control+n":case"Control+p":i.preventDefault()}switch((0,l.EL)(i.detail.originalEvent)){case"ArrowUp":case"ArrowDown":i.preventDefault();break;case"Enter":i.preventDefault()}}else{if(d)switch((0,l.EL)(i.detail.originalEvent)){case"Control+n":case"Control+p":case"Control+v":case"Alt+v":i.preventDefault()}switch((0,l.EL)(i.detail.originalEvent)){case"j":case"k":case"o":i.preventDefault();break;case"Enter":case`${y}+Enter`:i.preventDefault()}}}});function K(i){const n=i.modifierKey||i.altKey||i.ctrlKey||i.metaKey;(0,_.f)(i.currentTarget,"navigation:open",{modifierKey:n,shiftKey:i.shiftKey})||i.preventDefault()}a(K,"fireOpen"),(0,_.on)("click",".js-active-navigation-container .js-navigation-item",function(i){K(i)}),(0,_.on)("navigation:keyopen",".js-active-navigation-container .js-navigation-item",function(i){const n=i.currentTarget.classList.contains("js-navigation-open")?i.currentTarget:i.currentTarget.querySelector(".js-navigation-open");n instanceof HTMLAnchorElement?(i.detail.modifierKey?(window.open(n.href,"_blank"),window.focus()):n.dispatchEvent(new MouseEvent("click",{bubbles:!0,cancelable:!0}))&&n.click(),i.preventDefault()):K(i)});function R(i){const n=z();i!==n&&(n!==null&&Z(n),i==null||i.classList.add("js-active-navigation-container"))}a(R,"activate");function Z(i){i.classList.remove("js-active-navigation-container")}a(Z,"deactivate");const F=[];function V(i){const n=z();n&&F.push(n),R(i)}a(V,"push");function tt(i){Z(i),G(i);const n=F.pop();n&&R(n)}a(tt,"pop");function Q(i,n){const m=n||i,h=$(i)[0],C=m.closest(".js-navigation-item")||h;if(R(i),C instanceof HTMLElement){if(N(C,i))return;const M=(0,v.VZ)(C);I(M,C)}}a(Q,"navigation_focus");function G(i){const n=i.querySelectorAll(".js-navigation-item.navigation-focus");for(const m of n)m.classList.remove("navigation-focus")}a(G,"clear");function U(i,n){G(i),Q(i,n)}a(U,"refocus");function B(i,n){const m=$(n),h=m.indexOf(i),C=m[h-1];if(C){if(N(C,n))return;const M=(0,v.VZ)(C);J(n)==="page"?O(M,C):I(M,C)}}a(B,"cursorUp");function q(i,n){const m=$(n),h=m.indexOf(i),C=m[h+1];if(C){if(N(C,n))return;const M=(0,v.VZ)(C);J(n)==="page"?O(M,C):I(M,C)}}a(q,"cursorDown");function et(i,n){const m=$(n);let h=m.indexOf(i);const C=(0,v.VZ)(i);if(C==null)return;let S,M;for(;(S=m[h-1])&&(M=(0,v._C)(S,C))&&M.top>=0;)h--;if(S){if(N(S,n))return;O(C,S)}}a(et,"pageUp");function Y(i,n){const m=$(n);let h=m.indexOf(i);const C=(0,v.VZ)(i);if(C==null)return;let S,M;for(;(S=m[h+1])&&(M=(0,v._C)(S,C))&&M.bottom>=0;)h++;if(S){if(N(S,n))return;O(C,S)}}a(Y,"pageDown");function X(i,n=!1){(0,_.f)(i,"navigation:keyopen",{modifierKey:n})}a(X,"keyOpen");function N(i,n){return(0,_.f)(i,"navigation:focus")?(G(n),i.classList.add("navigation-focus"),!1):!0}a(N,"focusItem");function z(){return document.querySelector(".js-active-navigation-container")}a(z,"getActiveContainer");function $(i){const n=[];for(const m of i.querySelectorAll(".js-navigation-item"))m instanceof HTMLElement&&(0,f.Z)(m)&&n.push(m);return n}a($,"getItems");function J(i){return i.getAttribute("data-navigation-scroll")||"item"}a(J,"getScrollStyle");function O(i,n,m="smooth"){const h=(0,v._C)(n,i);!h||(h.bottom<=0?n.scrollIntoView({behavior:m,block:"start"}):h.top<=0&&n.scrollIntoView({behavior:m,block:"end"}))}a(O,"scrollPageTo");function I(i,n){const m=(0,v.oE)(n,i),h=(0,v._C)(n,i);if(!(m==null||h==null))if(h.bottom<=0&&document.body){const S=(i.offsetParent!=null?i.scrollHeight:document.body.scrollHeight)-(m.bottom+h.height);g(i,{top:S})}else h.top<=0&&g(i,{top:m.top})}a(I,"scrollItemTo")},63439:(k,P,b)=>{b.d(P,{L:()=>v,v:()=>l});var o=b(36738),_=b(44088);function v(g,f){const d=g||c();if(!d)return{};const y=d.querySelector(f||".js-notifications-list-item.navigation-focus");return y instanceof HTMLElement?{id:y.getAttribute("data-notification-id"),position:A(d).indexOf(y)}:{}}a(v,"getCurrentFocus");function l({id:g,position:f},d){const y=d||c();if(!(y instanceof HTMLElement))return;const T=A(y);let D;g&&(D=T.find(p=>p.getAttribute("data-notification-id")===g)),!D&&f!=null&&(D=T[Math.min(f,T.length-1)]),D instanceof HTMLElement&&(0,o.T_)(y,D)}a(l,"restoreFocus");function c(){return document.querySelector(".js-notifications-list .js-navigation-container")}a(c,"getNotificationsList");function A(g){return Array.from(g.querySelectorAll(".js-navigation-item")).filter(_.Z)}a(A,"getItems")},75914:(k,P,b)=>{b.d(P,{r:()=>o});function o(_,v){const l=_.createElement("template");return l.innerHTML=v,_.importNode(l.content,!0)}a(o,"parseHTML")},27316:(k,P,b)=>{b.d(P,{j:()=>o,u:()=>_});function o(v){const l=v.closest("form");if(!(l instanceof HTMLFormElement))return;let c=_(l);if(v.name){const A=v.matches("input[type=submit]")?"Submit":"",g=v.value||A;c||(c=document.createElement("input"),c.type="hidden",c.classList.add("is-submit-button-value"),l.prepend(c)),c.name=v.name,c.value=g}else c&&c.remove()}a(o,"persistSubmitButtonValue");function _(v){const l=v.querySelector("input.is-submit-button-value");return l instanceof HTMLInputElement?l:null}a(_,"findPersistedSubmitButtonValue")},27157:(k,P,b)=>{b.d(P,{w0:()=>o,RB:()=>_,qC:()=>v});class o{constructor(c){this.closed=!1,this.unsubscribe=()=>{c(),this.closed=!0}}}a(o,"Subscription");function _(l,c,A,g={capture:!1}){return l.addEventListener(c,A,g),new o(()=>{l.removeEventListener(c,A,g)})}a(_,"fromEvent");function v(...l){return new o(()=>{for(const c of l)c.unsubscribe()})}a(v,"compose")},44088:(k,P,b)=>{b.d(P,{Z:()=>_});function o(v){return v.offsetWidth<=0&&v.offsetHeight<=0}a(o,"hidden");function _(v){return!o(v)}a(_,"visible")},82388:(k,P,b)=>{var o=b(93673),_=Object.defineProperty,v=Object.getOwnPropertyDescriptor,l=a((t,e,u,r)=>{for(var s=r>1?void 0:r?v(e,u):e,E=t.length-1,w;E>=0;E--)(w=t[E])&&(s=(r?w(e,u,s):w(s))||s);return r&&s&&_(e,u,s),s},"__decorateClass");let c=a(class extends HTMLElement{constructor(){super();this.addEventListener("socket:message",this.update.bind(this))}update(t){const e=t.detail.data;this.link.setAttribute("aria-label",e.aria_label),this.link.setAttribute("data-ga-click",e.ga_click),this.modifier.setAttribute("class",e.span_class)}},"NotificationIndicatorElement");l([o.fA],c.prototype,"link",2),l([o.fA],c.prototype,"modifier",2),c=l([o.Ih],c);var A=Object.defineProperty,g=Object.getOwnPropertyDescriptor,f=a((t,e,u,r)=>{for(var s=r>1?void 0:r?g(e,u):e,E=t.length-1,w;E>=0;E--)(w=t[E])&&(s=(r?w(e,u,s):w(s))||s);return r&&s&&A(e,u,s),s},"notification_focus_indicator_element_decorateClass");let d=a(class extends HTMLElement{connectedCallback(){this.addEventListener("socket:message",t=>{const e=t.detail.data;this.link.setAttribute("aria-label",e.aria_label),this.link.setAttribute("data-ga-click",e.ga_click),this.modifier.setAttribute("class",e.span_class)})}toggleSidebar(){const t=new CustomEvent("notification-focus:toggle-sidebar",{bubbles:!0});this.dispatchEvent(t)}},"NotificationFocusIndicatorElement");f([o.fA],d.prototype,"link",2),f([o.fA],d.prototype,"modifier",2),d=f([o.Ih],d);var y=Object.defineProperty,T=Object.getOwnPropertyDescriptor,D=a((t,e,u,r)=>{for(var s=r>1?void 0:r?T(e,u):e,E=t.length-1,w;E>=0;E--)(w=t[E])&&(s=(r?w(e,u,s):w(s))||s);return r&&s&&y(e,u,s),s},"notification_focus_filters_element_decorateClass");let p=a(class extends HTMLElement{changeFilter(t){t.preventDefault(),this.detailsContainer.removeAttribute("open");const e=t.currentTarget;this.setFilterTitle(e.innerHTML),this.dispatchEvent(new CustomEvent("focus-mode-filter-change",{detail:{url:e.href}}))}setFilterTitle(t){this.filterTitle.innerHTML=t}},"NotificationFocusFiltersElement");D([o.fA],p.prototype,"detailsContainer",2),D([o.fA],p.prototype,"filterTitle",2),p=D([o.Ih],p);var L=b(63439),x=b(36738),j=b(16393),H=b(30762),K=b(11427),R=Object.defineProperty,Z=Object.getOwnPropertyDescriptor,F=a((t,e,u,r)=>{for(var s=r>1?void 0:r?Z(e,u):e,E=t.length-1,w;E>=0;E--)(w=t[E])&&(s=(r?w(e,u,s):w(s))||s);return r&&s&&R(e,u,s),s},"notification_focus_list_element_decorateClass");let V=a(class extends HTMLElement{connectedCallback(){(0,H.N7)(".js-notification-focus-list",()=>{this.setupPaginationObserver()}),(0,K.on)("pjax:end","#js-repo-pjax-container",()=>{this.toggleCurrentFocusedNotification()})}disconnectedCallback(){this.disconnectCurrentObserver()}deactivateNavigation(){(0,x.Sw)(this.container)}activateNavigation(){(0,x.VF)(this.container)}replaceContent(t){this.container.innerHTML="",this.container.appendChild(t),this.setupPaginationObserver()}onRemoveItem(t){var e,u,r;const s=t.detail.notificationId,E=(0,L.L)(this.container,".js-navigation-item.navigation-focus");(r=(u=(e=this.listElements)==null?void 0:e.find(w=>w.notificationId===s))==null?void 0:u.closest("li"))==null||r.remove(),this.listElements.length===0?(this.blankSlate.hidden=!1,this.list.hidden=!0):(0,L.v)(E,this.container)}toggleCurrentFocusedNotification(){for(const t of this.listElements){const e=window.location.href.includes(t.url());t.setFocusedState(e)}}setupPaginationObserver(){!!window.IntersectionObserver&&this.nextPageItem&&(this.currentObserver=new IntersectionObserver(t=>{!t[0].isIntersecting||(this.disconnectCurrentObserver(),this.loadNextPage())},{root:this.container,threshold:0}),this.currentObserver.observe(this.nextPageItem))}async loadNextPage(){if(!this.nextPageItem)return;const t=this.nextPageItem.getAttribute("data-next-page-url");if(t){this.nextPageItemSpinner.hidden=!1;const e=await(0,j.a)(document,t);this.nextPageItem.remove();const u=e.querySelectorAll("ul > li.focus-notification-item");for(const s of u)this.list.appendChild(s);const r=e.querySelector("ul > li.focus-pagination-next-item");r&&this.list.appendChild(r),this.setupPaginationObserver()}}disconnectCurrentObserver(){this.currentObserver&&this.currentObserver.disconnect()}},"NotificationFocusListElement");F([o.fA],V.prototype,"container",2),F([o.fA],V.prototype,"includeFragment",2),F([o.fA],V.prototype,"list",2),F([o.fA],V.prototype,"blankSlate",2),F([o.GO],V.prototype,"listElements",2),F([o.fA],V.prototype,"nextPageItem",2),F([o.fA],V.prototype,"nextPageItemSpinner",2),V=F([o.Ih],V);var tt=b(83895),Q=Object.defineProperty,G=Object.getOwnPropertyDescriptor,U=a((t,e,u,r)=>{for(var s=r>1?void 0:r?G(e,u):e,E=t.length-1,w;E>=0;E--)(w=t[E])&&(s=(r?w(e,u,s):w(s))||s);return r&&s&&Q(e,u,s),s},"notification_focus_list_item_element_decorateClass");let B=a(class extends HTMLElement{constructor(){super(...arguments);this.notificationId="",this.isUnread=!1}connectedCallback(){var t,e;(t=this.closest(".js-navigation-item"))==null||t.addEventListener("navigation:keydown",this.handleCustomKeybindings.bind(this)),(e=this.closest(".js-navigation-item"))==null||e.addEventListener("navigation:keyopen",this.handleKeyOpen.bind(this))}url(){var t;return(t=this.notificationLink)==null?void 0:t.href}handleCustomKeybindings(t){const e=t.detail;!(0,tt.Zf)(e.originalEvent)||(e.hotkey==="e"?this.doneForm.dispatchEvent(new Event("submit")):e.hotkey==="M"&&this.unsubscribeForm.dispatchEvent(new Event("submit")))}handleKeyOpen(){this.notificationLink.dispatchEvent(new MouseEvent("click",{bubbles:!0,cancelable:!0}))}setFocusedState(t){var e,u,r;t&&this.isUnread&&(this.isUnread=!1,(e=this.closest(".js-navigation-item"))==null||e.classList.remove("color-bg-default"),(u=this.closest(".js-navigation-item"))==null||u.classList.add("color-bg-subtle")),(r=this.closest(".js-navigation-item"))==null||r.classList.toggle("current-focused-item",t),this.notificationTitle.classList.toggle("text-bold",t||this.isUnread)}async runRemoveAction(t){t.preventDefault();const e=t.currentTarget,u=new FormData(e),r=e.method,s=e.action,{ok:E}=await fetch(s,{body:u,method:r});E&&this.dispatchEvent(new CustomEvent("focus-mode-remove-item",{bubbles:!0,detail:{notificationId:this.notificationId}}))}},"NotificationFocusListItemElement");U([o.Lj],B.prototype,"notificationId",2),U([o.Lj],B.prototype,"isUnread",2),U([o.fA],B.prototype,"doneForm",2),U([o.fA],B.prototype,"unsubscribeForm",2),U([o.fA],B.prototype,"notificationLink",2),U([o.fA],B.prototype,"notificationTitle",2),B=U([o.Ih],B);var q=Object.defineProperty,et=Object.getOwnPropertyDescriptor,Y=a((t,e,u,r)=>{for(var s=r>1?void 0:r?et(e,u):e,E=t.length-1,w;E>=0;E--)(w=t[E])&&(s=(r?w(e,u,s):w(s))||s);return r&&s&&q(e,u,s),s},"notification_focus_sidebar_element_decorateClass");let X=a(class extends HTMLElement{connectedCallback(){this.addEventListener("notification-focus:toggle-sidebar",this.toggleSidebar.bind(this),!0),window.localStorage.getItem("focus-sidebar-active")==="true"&&this.toggleSidebar()}toggleSidebar(){this.adjustSidebarPosition(),this.sidebar.classList.contains("active")?(this.listElement.deactivateNavigation(),this.sidebar.classList.remove("active"),window.localStorage.removeItem("focus-sidebar-active")):(this.listElement.activateNavigation(),this.sidebar.classList.add("active"),window.localStorage.setItem("focus-sidebar-active","true"))}async onFocusFilterChange(t){const e=t.detail;if(e.url){this.listElement.deactivateNavigation();const u=await(0,j.a)(document,e.url);this.listElement.replaceContent(u),this.listElement.activateNavigation()}}adjustSidebarPosition(){const t=document.querySelector("header[role=banner]");if(t){const e=t.offsetTop+t.offsetHeight;this.sidebar.style.top=`${e-10}px`}}},"NotificationFocusSidebarElement");Y([o.fA],X.prototype,"sidebar",2),Y([o.fA],X.prototype,"listElement",2),Y([o.fA],X.prototype,"filtersElement",2),X=Y([o.Ih],X);var N=b(42650),z=b(65889),$=Object.defineProperty,J=Object.getOwnPropertyDescriptor,O=a((t,e,u,r)=>{for(var s=r>1?void 0:r?J(e,u):e,E=t.length-1,w;E>=0;E--)(w=t[E])&&(s=(r?w(e,u,s):w(s))||s);return r&&s&&$(e,u,s),s},"notifications_list_subscription_form_element_decorateClass");let I=a(class extends HTMLElement{constructor(){super(...arguments);this.lastAppliedLabels={}}connectedCallback(){const t=this.querySelector(".js-label-subscriptions-load");t==null||t.addEventListener("loadend",()=>{this.subscriptionsLabels.length>0&&(this.updateCheckedState("custom"),this.updateMenuButtonCopy("custom"))})}async submitCustomForm(t){await this.submitForm(t),this.closeMenu()}async submitForm(t){t.preventDefault(),(0,N.H)();const e=t.currentTarget,u=new FormData(e),r=await self.fetch(e.action,{method:e.method,body:u,headers:{"X-Requested-With":"XMLHttpRequest",Accept:"application/json"}});if(!r.ok){(0,N.v)();return}const s=await r.json(),E=u.get("do");typeof E=="string"&&this.updateCheckedState(E),typeof E=="string"&&this.updateMenuButtonCopy(E),this.updateSocialCount(s.count),this.applyInputsCheckedPropertiesToAttributesForNextFormReset()}updateMenuButtonCopy(t){this.unwatchButtonCopy.hidden=!(t==="subscribed"||t==="custom"),this.stopIgnoringButtonCopy.hidden=t!=="ignore",this.watchButtonCopy.hidden=!(t!=="subscribed"&&t!=="custom"&&t!=="ignore")}applyInputsCheckedPropertiesToAttributesForNextFormReset(){for(const t of[...this.threadTypeCheckboxes])t.toggleAttribute("checked",t.checked)}updateCheckedState(t){for(const e of this.subscriptionButtons)e.setAttribute("aria-checked",e.value===t?"true":"false");if(t==="custom")this.customButton.setAttribute("aria-checked","true");else{this.customButton.setAttribute("aria-checked","false");for(const e of[...this.threadTypeCheckboxes])(0,z.Se)(e,!1);if(this.subscriptionsContainer!==void 0){for(let e=0;e<this.subscriptionsLabels.length;e++)this.subscriptionsLabels[e].remove();this.subscriptionsSubtitle!==void 0&&this.subscriptionsSubtitle.toggleAttribute("hidden",!1),this.subscriptionsContainer.innerHTML=""}}}updateSocialCount(t){this.socialCount&&(this.socialCount.textContent=t,this.socialCount.setAttribute("aria-label",`${this.pluralizeUsers(t)} watching this repository`))}pluralizeUsers(t){return parseInt(t)===1?"1 user is":`${t} users are`}handleDialogLabelToggle(t){const e=t.detail.wasChecked,u=t.detail.toggledLabelId,r=t.detail.templateLabelElementClone;if(e){for(let s=0;s<this.subscriptionsLabels.length;s++)if(this.subscriptionsLabels[s].getAttribute("data-label-id")===u){this.subscriptionsLabels[s].remove();break}}else r.removeAttribute("hidden"),r.setAttribute("data-targets","notifications-list-subscription-form.subscriptionsLabels"),this.subscriptionsContainer.appendChild(r)}openCustomDialog(t){t.preventDefault(),t.stopPropagation(),this.menu.toggleAttribute("hidden",!0),this.enableApplyButtonAndCheckbox(),this.saveCurrentLabelsState(),this.customDialog.toggleAttribute("hidden",!1),setTimeout(()=>{var e;(e=this.customDialog.querySelector("input[type=checkbox][autofocus]"))==null||e.focus()},0)}enableApplyButtonAndCheckbox(){this.customDialog.querySelectorAll('[data-type="label"]:not([hidden])').length>0&&(this.customSubmit.removeAttribute("disabled"),this.threadTypeCheckboxes[0].checked=!0)}closeCustomDialog(t){t.preventDefault(),t.stopPropagation(),this.menu.toggleAttribute("hidden",!1),this.customDialog.toggleAttribute("hidden",!0),setTimeout(()=>{this.customButton.focus()},0)}resetFilterLabelsDialog(t){t.preventDefault(),t.stopPropagation();for(let e=0;e<this.subscriptionsLabels.length;e++){const u=this.subscriptionsLabels[e].getAttribute("data-label-id");for(let r=0;r<this.dialogLabelItems.length;r++)if(this.dialogLabelItems[r].labelId===u){this.dialogLabelItems[r].setCheckedForDropdownLabel(!1);break}}for(let e=0;e<Object.keys(this.lastAppliedLabels).length;e++){const u=Object.keys(this.lastAppliedLabels)[e];for(let r=0;r<this.dialogLabelItems.length;r++)if(this.dialogLabelItems[r].labelId===u){this.dialogLabelItems[r].setCheckedForDropdownLabel(!0);break}}this.subscriptionsContainer.replaceChildren(...Object.values(this.lastAppliedLabels)),this.closeFilterLabelsDialog(t)}openFilterLabelsDialog(t){t.preventDefault(),t.stopPropagation(),this.saveCurrentLabelsState(),this.customDialog.toggleAttribute("hidden",!0),this.filterLabelsDialog.toggleAttribute("hidden",!1),setTimeout(()=>{var e;(e=this.filterLabelsDialog.querySelector("input[type=checkbox][autofocus]"))==null||e.focus()},0)}closeFilterLabelsDialog(t){t.preventDefault(),t.stopPropagation(),this.menu.toggleAttribute("hidden",!0),this.customDialog.toggleAttribute("hidden",!1),this.filterLabelsDialog.toggleAttribute("hidden",!0)}applyFilterLabelsDialog(t){t.preventDefault(),t.stopPropagation(),this.saveCurrentLabelsState(),this.hideFilterSubtitle(),this.enableIssuesCheckbox(),this.closeFilterLabelsDialog(t)}enableIssuesCheckbox(){const t=Object.keys(this.lastAppliedLabels).length>0;t&&this.threadTypeCheckboxes.length>0&&(this.threadTypeCheckboxes[0].checked=t),this.threadTypeCheckboxesUpdated()}hideFilterSubtitle(){const t=Object.keys(this.lastAppliedLabels).length>0;this.subscriptionsSubtitle.toggleAttribute("hidden",t)}saveCurrentLabelsState(){this.lastAppliedLabels={},this.labelInputs.innerHTML="";for(let t=0;t<this.subscriptionsLabels.length;t++){const e=this.subscriptionsLabels[t].getAttribute("data-label-id");e&&(this.lastAppliedLabels[e]=this.subscriptionsLabels[t].cloneNode(!0),this.appendLabelToFormInput(e))}}appendLabelToFormInput(t){const e=document.createElement("input");e.setAttribute("type","hidden"),e.setAttribute("name","labels[]"),e.setAttribute("value",t),this.labelInputs.appendChild(e)}detailsToggled(){this.menu.toggleAttribute("hidden",!1),this.customDialog.toggleAttribute("hidden",!0)}submitCustom(t){t.preventDefault(),this.details.toggleAttribute("open",!1)}threadTypeCheckboxesUpdated(){const t=!this.threadTypeCheckboxes.some(e=>e.checked);this.customSubmit.disabled=t}closeMenu(){this.details.toggleAttribute("open",!1)}},"NotificationsListSubscriptionFormElement");O([o.fA],I.prototype,"details",2),O([o.fA],I.prototype,"menu",2),O([o.fA],I.prototype,"customButton",2),O([o.fA],I.prototype,"customDialog",2),O([o.fA],I.prototype,"filterLabelsDialog",2),O([o.GO],I.prototype,"subscriptionButtons",2),O([o.GO],I.prototype,"subscriptionsLabels",2),O([o.fA],I.prototype,"labelInputs",2),O([o.fA],I.prototype,"subscriptionsSubtitle",2),O([o.fA],I.prototype,"socialCount",2),O([o.fA],I.prototype,"unwatchButtonCopy",2),O([o.fA],I.prototype,"stopIgnoringButtonCopy",2),O([o.fA],I.prototype,"watchButtonCopy",2),O([o.GO],I.prototype,"threadTypeCheckboxes",2),O([o.fA],I.prototype,"customSubmit",2),O([o.fA],I.prototype,"subscriptionsContainer",2),O([o.GO],I.prototype,"dialogLabelItems",2),I=O([o.Ih],I);var i=Object.defineProperty,n=Object.getOwnPropertyDescriptor,m=a((t,e,u,r)=>{for(var s=r>1?void 0:r?n(e,u):e,E=t.length-1,w;E>=0;E--)(w=t[E])&&(s=(r?w(e,u,s):w(s))||s);return r&&s&&i(e,u,s),s},"notifications_team_subscription_form_element_decorateClass");let h=a(class extends HTMLElement{closeMenu(){this.details.toggleAttribute("open",!1)}},"NotificationsTeamSubscriptionFormElement");m([o.fA],h.prototype,"details",2),h=m([o.Ih],h);var C=Object.defineProperty,S=Object.getOwnPropertyDescriptor,M=a((t,e,u,r)=>{for(var s=r>1?void 0:r?S(e,u):e,E=t.length-1,w;E>=0;E--)(w=t[E])&&(s=(r?w(e,u,s):w(s))||s);return r&&s&&C(e,u,s),s},"notifications_subscriptions_dialog_label_item_decorateClass");let W=a(class extends HTMLElement{toggleDropdownLabel(t){if(t.preventDefault(),t.stopPropagation(),this.label){const e=this.label.getAttribute("aria-checked")==="true";this.setCheckedForDropdownLabel(!e),this.dispatchEvent(new CustomEvent("notifications-dialog-label-toggled",{detail:{wasChecked:e,toggledLabelId:this.labelId,templateLabelElementClone:this.hiddenLabelTemplate.cloneNode(!0)},bubbles:!0}))}}setCheckedForDropdownLabel(t){this.label.setAttribute("aria-checked",t.toString())}},"NotificationsDialogLabelItemElement");M([o.fA],W.prototype,"label",2),M([o.fA],W.prototype,"hiddenLabelTemplate",2),M([o.fA],W.prototype,"hiddenCheckboxInput",2),M([o.Lj],W.prototype,"labelId",2),W=M([o.Ih],W)}},k=>{var P=a(o=>k(k.s=o),"__webpack_exec__");k.O(0,["vendors-node_modules_selector-observer_dist_index_esm_js","vendors-node_modules_delegated-events_dist_index_js-node_modules_github_catalyst_lib_index_js-6e358f"],()=>P(82388));var b=k.O()}]);})();

//# sourceMappingURL=notifications-global-c396c93e8ae016febc2c2eea7bb6cd1fc692ae8cdde50a2cc3b3e37bee2780c5f771ef657b83ed6af0e5628fcdf41edbb9bdcee82e7d805a832e6ac2b1528784.js.map