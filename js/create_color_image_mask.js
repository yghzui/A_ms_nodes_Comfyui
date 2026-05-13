import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "CreateColorImageAndMask",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "CreateColorImageAndMask") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const colorInput = document.createElement("input");
                colorInput.type = "color";
                colorInput.value = "#FF0000";
                
                colorInput.style.position = "absolute";
                colorInput.style.left = "0px";
                colorInput.style.top = "24px";
                colorInput.style.width = "100%";
                colorInput.style.height = "24px";
                
                this.widgets.find(w => w.name === "color").inputEl.style.display = "none";
                this.widgets.find(w => w.name === "color").element.appendChild(colorInput);
                
                colorInput.addEventListener("input", (e) => {
                    this.widgets.find(w => w.name === "color").value = e.target.value;
                });
                
                return result;
            };
        }
    }
}); 