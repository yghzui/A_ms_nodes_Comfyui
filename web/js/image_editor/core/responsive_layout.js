// 本文件由主项目 tools/sync_image_editor_core.py 自动生成，禁止手改。
// 源文件: web_apps/static/image_editor/responsive_layout.js
export const MOBILE_BREAKPOINT = 900;

export const syncCompactPanelState = (ctx, open) => {
    const nextOpen = !!(ctx.isCompactLayout && open);
    ctx.rightBar.style.transform = ctx.isCompactLayout && !nextOpen ? "translateX(110%)" : "translateX(0)";
    ctx.rightBar.style.pointerEvents = ctx.isCompactLayout && !nextOpen ? "none" : "auto";
    ctx.mobilePanelBackdrop.style.display = ctx.isCompactLayout && nextOpen ? "block" : "none";
    ctx.mobilePanelToggleBtn.textContent = nextOpen ? "✕ 关闭" : "⚙ 参数";
    return nextOpen;
};

export const applyResponsiveLayout = (ctx) => {
    const isCompactLayout = ctx.windowWidth <= MOBILE_BREAKPOINT || ctx.windowHeight <= 720;

    ctx.topBar.style.height = isCompactLayout ? "46px" : "50px";
    ctx.topBar.style.padding = isCompactLayout ? "0 8px" : "0 20px";
    ctx.topBar.style.gap = isCompactLayout ? "6px" : "10px";
    ctx.topBar.style.overflowX = isCompactLayout ? "auto" : "visible";
    ctx.topBar.style.flexWrap = "nowrap";
    ctx.title.style.fontSize = isCompactLayout ? "14px" : "";
    ctx.title.style.marginRight = isCompactLayout ? "6px" : "20px";
    ctx.spacer.style.display = isCompactLayout ? "none" : "block";
    ctx.closeBtn.style.marginLeft = isCompactLayout ? "0" : "10px";
    ctx.closeBtn.style.display = isCompactLayout ? "none" : "flex";

    ctx.topActionButtons.forEach((btn) => {
        const labels = ctx.topActionButtonLabels.get(btn);
        if (labels) btn.textContent = isCompactLayout ? labels.compact : labels.full;
        btn.style.padding = isCompactLayout ? "6px 8px" : (btn === ctx.closeBtn ? "4px 8px" : "6px 12px");
        btn.style.fontSize = isCompactLayout ? "12px" : (btn === ctx.closeBtn ? "18px" : "14px");
        btn.style.minHeight = isCompactLayout ? "34px" : "auto";
        btn.style.whiteSpace = "nowrap";
    });

    ctx.leftBar.style.width = isCompactLayout ? "auto" : "60px";
    ctx.leftBar.style.position = isCompactLayout ? "absolute" : "relative";
    ctx.leftBar.style.left = isCompactLayout ? "8px" : "";
    ctx.leftBar.style.right = isCompactLayout ? "8px" : "";
    ctx.leftBar.style.bottom = isCompactLayout ? "max(10px, env(safe-area-inset-bottom))" : "";
    ctx.leftBar.style.top = isCompactLayout ? "auto" : "";
    ctx.leftBar.style.zIndex = isCompactLayout ? "6" : "";
    ctx.leftBar.style.flexDirection = isCompactLayout ? "row" : "column";
    ctx.leftBar.style.justifyContent = isCompactLayout ? "center" : "flex-start";
    ctx.leftBar.style.alignItems = "center";
    ctx.leftBar.style.padding = isCompactLayout ? "8px 10px" : "15px 0 0 0";
    ctx.leftBar.style.gap = isCompactLayout ? "8px" : "15px";
    ctx.leftBar.style.borderRight = isCompactLayout ? "none" : "1px solid #444";
    ctx.leftBar.style.borderRadius = isCompactLayout ? "14px" : "0";
    ctx.leftBar.style.background = isCompactLayout ? "rgba(42, 42, 42, 0.92)" : "#2a2a2a";
    ctx.leftBar.style.backdropFilter = isCompactLayout ? "blur(8px)" : "none";
    ctx.leftBar.style.overflowX = isCompactLayout ? "auto" : "hidden";
    ctx.leftBar.style.overflowY = "hidden";
    ctx.leftBar.style.boxShadow = isCompactLayout ? "0 10px 25px rgba(0, 0, 0, 0.35)" : "none";

    Object.values(ctx.toolBtns).forEach((btn) => {
        btn.style.width = isCompactLayout ? "46px" : "40px";
        btn.style.height = isCompactLayout ? "46px" : "40px";
        btn.style.fontSize = isCompactLayout ? "22px" : "20px";
        btn.style.flex = isCompactLayout ? "0 0 auto" : "";
    });
    ctx.infoLabel.style.display = isCompactLayout ? "none" : "block";
    ctx.workArea.style.paddingBottom = isCompactLayout ? "82px" : "0";

    ctx.rightBar.style.position = isCompactLayout ? "absolute" : "relative";
    ctx.rightBar.style.top = isCompactLayout ? "0" : "";
    ctx.rightBar.style.right = isCompactLayout ? "0" : "";
    ctx.rightBar.style.bottom = isCompactLayout ? "0" : "";
    ctx.rightBar.style.zIndex = isCompactLayout ? "8" : "";
    ctx.rightBar.style.width = isCompactLayout ? `${Math.min(Math.floor(ctx.windowWidth * 0.84), 320)}px` : "260px";
    ctx.rightBar.style.maxWidth = isCompactLayout ? "84vw" : "";
    ctx.rightBar.style.boxShadow = isCompactLayout ? "-12px 0 30px rgba(0, 0, 0, 0.4)" : "none";
    ctx.rightBar.style.padding = isCompactLayout ? "16px 14px calc(24px + env(safe-area-inset-bottom)) 14px" : "20px";
    ctx.rightBar.style.borderLeft = isCompactLayout ? "1px solid #555" : "1px solid #444";
    ctx.rightBar.style.transition = "transform 0.2s ease";
    ctx.rightBar.style.gap = isCompactLayout ? "18px" : "25px";
    ctx.mobilePanelToggleBtn.style.display = isCompactLayout ? "flex" : "none";
    ctx.mobilePanelHeader.style.display = isCompactLayout ? "flex" : "none";

    const mobilePanelOpen = syncCompactPanelState({ ...ctx, isCompactLayout }, isCompactLayout ? ctx.mobilePanelOpen : false);
    if (!isCompactLayout) {
        ctx.mobilePanelBackdrop.style.display = "none";
        ctx.rightBar.style.pointerEvents = "auto";
        ctx.rightBar.style.transform = "translateX(0)";
    }
    return { isCompactLayout, mobilePanelOpen };
};
