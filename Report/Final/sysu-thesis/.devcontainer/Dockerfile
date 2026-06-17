FROM texlive/texlive:TL2024-historic

RUN echo "deb http://mirror.sysu.edu.cn/debian/ forky main contrib non-free" > /etc/apt/sources.list \
    && echo "deb http://mirrors.ustc.edu.cn/debian/ forky main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ forky main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb http://mirror.sysu.edu.cn/debian-security/ forky-security main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.ustc.edu.cn/debian-security/ forky-security main contrib non-free" >> /etc/apt/sources.list \
    && echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian-security/ forky-security main contrib non-free" >> /etc/apt/sources.list \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends ttf-mscorefonts-installer make fonts-arphic-ukai fonts-noto-cjk-extra \
    && fc-cache \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && touch /.sysu-thesis.devcontainer

ENV SHELL /bin/bash

CMD ["/bin/bash"]