struct InputData {
	float Position;
	int Index_x;
	int Index_y;
};

RWStructuredBuffer<InputData> data : register(u0);

cbuffer vars : register(b0)
{
	float uTime;
};

struct PSInput
{
	float2 UV : TEXCOORD;
	uint instanceID : INSTANCEID;
};


Texture2D tex : register(t0);
SamplerState smp : register(s0);

float4 main(PSInput pin) : SV_TARGET {
	float x_loc = 64.0 / 256.0;
	float y_loc = 64.0 / 192.0;
	
	int x_index = data[pin.instanceID].Index_x;
	int y_index = data[pin.instanceID].Index_y;
	
	float2 uv = pin.UV;
	uv.x = (uv.x * x_loc) + (x_index * x_loc);
	uv.y = (uv.y * y_loc) + (y_index * y_loc);

    return tex.Sample(smp, uv);
}
