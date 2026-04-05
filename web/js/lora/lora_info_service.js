// LoRA信息服务
export class LoraInfoService {
    constructor() {
        this.cache = new Map();
    }

    async getInfo(loraName, force = false, includeDetails = true) {
        const cacheKey = `${loraName}_${includeDetails}`;
        
        if (!force && this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            // 获取LoRA信息
            const response = await fetch(`/loras?format=details`);
            const loras = await response.json();
            
            const loraInfo = loras.find(lora => lora.file === loraName);
            
            if (loraInfo) {
                const info = {
                    name: loraInfo.file,
                    size: loraInfo.size,
                    description: loraInfo.description || "无描述",
                    strengthMin: -2.0,
                    strengthMax: 2.0,
                    // 可以添加更多信息
                };
                
                this.cache.set(cacheKey, info);
                return info;
            }
            
            return null;
        } catch (error) {
            console.error("获取LoRA信息失败:", error);
            return null;
        }
    }

    clearCache() {
        this.cache.clear();
    }
}

export const LORA_INFO_SERVICE = new LoraInfoService(); 