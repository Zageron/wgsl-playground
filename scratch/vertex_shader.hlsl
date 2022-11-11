struct InputData {
	float Position;
	int Index_x;
	int Index_y;
};

RWStructuredBuffer<InputData> data : register(u0);

cbuffer vars : register(b0)
{
	float2 uResolution;
	float uTime;
};

struct VSInput
{
	float2 Position : POSITION;
	float2 UV : TEXCOORD;
	uint instanceId : SV_InstanceID;
};

struct VSOutput
{
	float4 Position : SV_POSITION;
	float2 UV : TEXCOORD;
};

VSOutput main(VSInput vin) {
	VSOutput vout = (VSOutput)0;
	
	float2 tex_size = float2(64, 64);
	float2 temp_pos = vin.Position;
	float2 pos = temp_pos / uResolution * tex_size;
	
	
    float2 j = pos + float2(0, data[vin.instanceId].Position);
    float2 uv = floor(j);
	float2 pos2 = uv += 1.0 - clamp((1.0 - frac(j)) * 1, 0.0, 1.0);
	vout.Position = float4(pos2, 0.0, 1.0);
	vout.UV = vin.UV;
	
	return vout;
}
