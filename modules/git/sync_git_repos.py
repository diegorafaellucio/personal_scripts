#!/usr/bin/env python3
"""
Script para sincronizar repositórios Git
- Baixa atualizações de 'origin'
- Sincroniza com 'second'
"""
import os
import subprocess
import sys
from typing import List


def run_git_command(repo_path: str, command: List[str]) -> subprocess.CompletedProcess:
    """Executa um comando git no repositório especificado"""
    try:
        result = subprocess.run(
            ["git"] + command,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"ERRO no repositório {repo_path}: {e}")
        print(f"Saída de erro: {e.stderr}")
        return e


def is_git_repo(path: str) -> bool:
    """Verifica se o diretório é um repositório Git"""
    git_dir = os.path.join(path, ".git")
    return os.path.isdir(git_dir)


def has_remote(repo_path: str, remote_name: str) -> bool:
    """Verifica se o repositório tem o remote especificado"""
    try:
        result = run_git_command(repo_path, ["remote"])
        return remote_name in result.stdout.split()
    except:
        return False


def sync_repository(repo_path: str) -> None:
    """Sincroniza um repositório Git (pull de origin, push para second)"""
    repo_name = os.path.basename(repo_path)
    
    print(f"\n{'='*80}\nProcessando repositório: {repo_name} ({repo_path})\n{'='*80}")
    
    # Verifica se tem os remotes necessários
    if not has_remote(repo_path, "origin"):
        print(f"Repositório {repo_name} não possui o remote 'origin'. Pulando.")
        return
    
    # Obtém a lista de branches
    branches_result = run_git_command(repo_path, ["branch"])
    if isinstance(branches_result, subprocess.CalledProcessError):
        return
    
    # Extrai os nomes das branches (remove o * e espaços)
    branches = [
        branch.strip().replace("* ", "") 
        for branch in branches_result.stdout.splitlines()
    ]
    
    # Puxa todas as branches de origin
    print(f"Atualizando de 'origin' todas as branches...")
    fetch_result = run_git_command(repo_path, ["fetch", "--all"])
    if isinstance(fetch_result, subprocess.CalledProcessError):
        return
    
    # Para cada branch local, puxa de origin e tenta enviar para second
    for branch in branches:
        print(f"\nProcessando branch: {branch}")
        
        # Muda para a branch
        checkout_result = run_git_command(repo_path, ["checkout", branch])
        if isinstance(checkout_result, subprocess.CalledProcessError):
            continue
        
        # Puxa de origin
        print(f"Puxando branch {branch} de origin...")
        pull_result = run_git_command(repo_path, ["pull", "origin", branch])
        if isinstance(pull_result, subprocess.CalledProcessError):
            continue
        else:
            print(pull_result.stdout)
        
        # Verifica se tem o remote second
        if has_remote(repo_path, "second"):
            # Envia para second
            print(f"Enviando branch {branch} para second...")
            push_result = run_git_command(repo_path, ["push", "second", branch])
            if isinstance(push_result, subprocess.CalledProcessError):
                continue
            else:
                print(push_result.stdout)
        else:
            print(f"Repositório {repo_name} não possui o remote 'second'. Pulando sincronização.")


def find_and_sync_repos(root_dir: str) -> None:
    """Procura e sincroniza todos os repositórios Git em um diretório"""
    print(f"Buscando repositórios Git em: {root_dir}")
    
    for company in os.listdir(root_dir):
        company_path = os.path.join(root_dir, company)
        
        # Pula se não for diretório
        if not os.path.isdir(company_path):
            continue
            
        print(f"\n\n{'#'*80}\nEmpresa: {company}\n{'#'*80}")
        
        # Procura projetos na pasta da empresa
        for project in os.listdir(company_path):
            project_path = os.path.join(company_path, project)
            
            # Pula se não for diretório
            if not os.path.isdir(project_path):
                continue
                
            # Se for um repositório Git, sincroniza
            if is_git_repo(project_path):
                sync_repository(project_path)
            else:
                # Pode ser um diretório que contém vários repositórios
                for subdir in os.listdir(project_path):
                    subdir_path = os.path.join(project_path, subdir)
                    if os.path.isdir(subdir_path) and is_git_repo(subdir_path):
                        sync_repository(subdir_path)


if __name__ == "__main__":
    # Define o diretório raiz (pode ser sobrescrito por argumento de linha de comando)
    root_dir = "/home/diego/projects"
    
    # Verifica se foi passado um argumento com o diretório raiz
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    
    # Verifica se o diretório existe
    if not os.path.isdir(root_dir):
        print(f"Diretório {root_dir} não encontrado!")
        sys.exit(1)
    
    # Procura e sincroniza todos os repositórios
    find_and_sync_repos(root_dir)
    
    print("\nSincronização concluída!")
