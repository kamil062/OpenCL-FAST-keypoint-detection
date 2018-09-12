__kernel void fast_detect(
    read_only image2d_t image,
    const uint width,
    const uint height,
    const uint N,
    const float threshold,
    __global int *is_keypoint,
    __global int *scores
)
{
    const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    uint pos_1d = get_global_id(1) * width + get_global_id(0);

    uint4 p = read_imageui(image, sampler, (int2)(pos.x, pos.y));
    int p_darker = p.x - threshold;
    int p_brighter = p.x + threshold;

    int x_shifts[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
    int y_shifts[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};

    uint4 _p[16]; 

    int i;
    for(i = 0; i < 16; i++)
        _p[i] = read_imageui(image, sampler, (int2)(pos.x + x_shifts[i], pos.y + y_shifts[i]));

    if(_p[0].x < p_darker && _p[8].x < p_darker){
        if(_p[4].x < p_darker && _p[12].x < p_darker){

            uint sum_darker = 0;

            for(i = 0; i < 16; i++)
                sum_darker += _p[i].x < p_darker ? 1 : 0;

            if(sum_darker >= N){
                is_keypoint[pos_1d] = 1;

                int score = 0;

                for(i = 0; i < 16; i++)
                    score += abs(p.x - _p[i].x);

                scores[pos_1d] = score;

                return;
            }
        }
    }
    
    if(_p[0].x > p_brighter && _p[8].x > p_brighter){
        if(_p[4].x > p_brighter && _p[12].x > p_brighter){

            uint sum_brighter = 0;

            for(i = 0; i < 16; i++)
                sum_brighter += _p[i].x > p_brighter ? 1 : 0;

            if(sum_brighter >= N){
                is_keypoint[pos_1d] = 1;

                int score = 0;

                for(i = 0; i < 16; i++)
                    score += abs(p.x - _p[i].x);

                scores[pos_1d] = score;
            }
        }
    }
}